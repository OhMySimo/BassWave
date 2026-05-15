# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Ghosting / sidechain-style volume artefacts.

Models the kind of level wobble that comes from:
  * sidechain ducking from kick/snare into the bass channel
  * imperfect phase relationship with another instrument (kick, piano, guitar)
    causing transient cancellation/reinforcement
  * gain riding by a careless mixer

Implementation:
  1. Generate sparse Bernoulli impulse events at the audio sample rate.
     ~`events_per_sec` events on average per second.
  2. Each event has random sign (±) and magnitude up to `max_db_swing` dB.
  3. Convolve impulses with an asymmetric attack/release kernel:
        * fast attack: linear rise over `attack_ms`
        * slow release: exponential decay over `release_ms`
     This gives the characteristic "duck and slowly recover" envelope.
  4. Convert dB envelope to linear gain, multiply audio.

Per-example randomness:
  * Event positions: independent Bernoulli per sample, per example.
  * Event signs: independent ±1 per sample, per example.
  * Event magnitudes: uniform [0, max_db_swing], per sample, per example.

Per-batch randomness (kept fixed per call to avoid variable-size tf ops):
  * Attack and release times.
"""

import tensorflow.compat.v2 as tf

from ddsp.training.degradations.base import BaseDegradation, db_to_lin


class Ghosting(BaseDegradation):
  """Sparse-event volume modulation with attack/release envelope shaping."""

  def __init__(self,
               wet_dry_pct: float = 50.0,
               sample_rate: int = 44100,
               max_db_swing: float = 3.0,
               events_per_sec: float = 20.0,
               attack_ms_min: float = 0.5,
               attack_ms_max: float = 5.0,
               release_ms_min: float = 20.0,
               release_ms_max: float = 100.0,
               name: str = 'ghosting',
               **kwargs):
    super().__init__(wet_dry_pct=wet_dry_pct, sample_rate=sample_rate,
                     name=name, **kwargs)
    self.max_db_swing = float(max_db_swing)
    self.events_per_sec = float(events_per_sec)
    self.attack_ms_range = (attack_ms_min, attack_ms_max)
    self.release_ms_range = (release_ms_min, release_ms_max)
    # Pre-compute the maximum kernel size for static shape.
    self._max_attack = int(attack_ms_max * 1e-3 * sample_rate) + 1
    self._max_release = int(release_ms_max * 1e-3 * sample_rate) + 1
    self._kernel_size = self._max_attack + self._max_release

  def _build_envelope_kernel(self):
    """Sample one attack-release kernel for this call (shared by the batch).

    Variable kernel sizes are awkward in tf.function (would need
    `tf.signal.frame` or padded conv). We keep the kernel size FIXED at the
    maximum and zero out the unused portion. This is graph-friendly and the
    extra zeros cost essentially nothing in the conv.
    """
    sr = float(self.sample_rate)
    a_ms = tf.random.uniform([], self.attack_ms_range[0],
                             self.attack_ms_range[1])
    r_ms = tf.random.uniform([], self.release_ms_range[0],
                             self.release_ms_range[1])
    a_n = tf.cast(a_ms * 1e-3 * sr, tf.int32)
    r_n = tf.cast(r_ms * 1e-3 * sr, tf.int32)

    # Build mask + linear ramp + exp decay over the FIXED-SIZE buffer.
    pos = tf.range(self._kernel_size, dtype=tf.float32)

    # Attack ramp: 0 → 1 over [0, a_n], else 0 (after the peak the rise stops).
    a_mask = tf.cast(pos < tf.cast(a_n, tf.float32), tf.float32)
    rise = tf.where(
        a_mask > 0.5,
        pos / tf.maximum(tf.cast(a_n, tf.float32), 1.0),
        tf.zeros_like(pos))

    # Release decay: starts at index a_n with value 1 and decays exponentially.
    r_pos = pos - tf.cast(a_n, tf.float32)
    r_mask = tf.cast(
        (r_pos >= 0) & (r_pos < tf.cast(r_n, tf.float32)), tf.float32)
    fall = r_mask * tf.exp(
        -3.0 * r_pos / tf.maximum(tf.cast(r_n, tf.float32), 1.0))

    # Composite: rise during attack phase, exp decay during release phase.
    kernel = rise + fall
    # Normalise peak to 1 so the kernel itself doesn't change overall gain.
    kernel = kernel / (tf.reduce_max(kernel) + 1e-6)
    return kernel  # [kernel_size]

  def _degrade(self, audio):
    batch = tf.shape(audio)[0]
    n = tf.shape(audio)[1]

    # 1. Sparse random impulse events.
    p_event = self.events_per_sec / float(self.sample_rate)
    bernoulli = tf.cast(
        tf.random.uniform([batch, n]) < p_event, tf.float32)
    sign = tf.where(
        tf.random.uniform([batch, n]) < 0.5,
        tf.ones([batch, n]),
        -tf.ones([batch, n]))
    magnitude = tf.random.uniform(
        [batch, n], 0.0, self.max_db_swing)
    impulses_db = bernoulli * sign * magnitude            # [B, n] in dB

    # 2. Convolve with attack-release kernel.
    kernel = self._build_envelope_kernel()                 # [K]
    # tf.nn.conv1d wants [B, T, in_ch] and [W, in_ch, out_ch].
    impulses_4d = impulses_db[:, :, tf.newaxis]            # [B, n, 1]
    kernel_3d = kernel[:, tf.newaxis, tf.newaxis]          # [K, 1, 1]
    env_db = tf.nn.conv1d(
        impulses_4d, kernel_3d, stride=1, padding='SAME')
    env_db = tf.squeeze(env_db, axis=-1)                   # [B, n]

    # 3. dB → linear gain, apply.
    gain = db_to_lin(env_db)
    return audio * gain
