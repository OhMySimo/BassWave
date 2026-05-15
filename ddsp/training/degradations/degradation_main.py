# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Degradation pipeline — chooses K∈[min_n_apply, max_n_apply] degradations
per training step, applies them sequentially with per-instance wet/dry
control.

User-facing knobs (configurable via gin):
  * Per-degradation wet/dry (0..100): independently controls the strength of
    each of the 6 degradations.
  * `min_n_apply` / `max_n_apply`: range from which K is drawn each call
    (default 2 / 6 = the user's spec).
  * `clean_passthrough_prob`: fraction of examples that pass through with NO
    degradation (essential — without this the model never sees clean inputs
    in training and forgets how to behave on them).
  * `wet_schedule_steps` + `wet_schedule_start_step`: linearly ramp each
    layer's wet from its start value to 100 % over this many training
    steps. Schedule starts FRESH at each training session (the counter is
    not checkpointed; see _sched_step note below).
  * `n_apply_schedule_steps` (NEW): linearly ramp the K range from
    (min_n_apply_start, max_n_apply_start) to (min_n_apply, max_n_apply)
    over this many training steps. Lets you start with 1-2 layers per
    batch and grow to 4-6 by the end of curriculum.

Per-batch random selection (one set of K degradations per batch). Within the
selected set, every layer's *parameters* are randomised per-example, so the
batch is far from homogeneous.

Order of application is fixed: phasing → ghosting → wrong_eq → compression
→ dynamic_eq → bandwidth_limit. EQ-style spectral changes happen before
non-linearities (compression, clipping), mirroring real signal chains.
"""

import tensorflow.compat.v2 as tf

from ddsp.training.degradations.aggressive_compression import (
    AggressiveCompression)
from ddsp.training.degradations.bandwidth_limit import BandwidthLimit
from ddsp.training.degradations.dynamic_eq import DynamicEQ
from ddsp.training.degradations.ghosting import Ghosting
from ddsp.training.degradations.phasing import Phasing
from ddsp.training.degradations.wrong_eq import WrongEQ


class DegradationPipeline(tf.keras.layers.Layer):
  """Random K-of-6 degradation pipeline for BassWave training augmentation."""

  def __init__(self,
               sample_rate: int = 44100,
               # Per-degradation wet/dry (0..100). These are the STARTING
               # values; if wet_schedule_steps > 0 they increase linearly
               # to 100 over wet_schedule_steps training steps.
               phasing_wet: float = 50.0,
               ghosting_wet: float = 50.0,
               wrong_eq_wet: float = 50.0,
               compression_wet: float = 50.0,
               dynamic_eq_wet: float = 50.0,
               bandwidth_wet: float = 50.0,
               # Pipeline-level controls (FINAL values after schedule).
               clean_passthrough_prob: float = 0.10,
               min_n_apply: int = 2,
               max_n_apply: int = 6,
               # n_apply schedule START values. If n_apply_schedule_steps > 0
               # we ramp (min_n_apply_start, max_n_apply_start) →
               # (min_n_apply, max_n_apply) over the schedule. Defaults
               # equal the FINAL values = no ramp (legacy behaviour).
               min_n_apply_start: int = -1,
               max_n_apply_start: int = -1,
               n_apply_schedule_steps: int = 0,
               preserve_loudness: bool = True,
               max_loudness_gain_db: float = 12.0,
               # Wet schedule: linearly ramp each wet from its start value
               # to 100 % over this many training steps. 0 = disabled.
               wet_schedule_steps: int = 0,
               # Schedule start step (used as the initial value of the
               # internal counter at construction time). NOTE: the counter
               # is intentionally NOT checkpointed (see __init__ for why),
               # so this value is the "fresh-session" starting point. To
               # resume mid-schedule across runs you must set this to the
               # step at which the schedule was last paused.
               wet_schedule_start_step: int = 0,
               name: str = 'degradation_pipeline',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.sample_rate = int(sample_rate)
    self.clean_passthrough_prob = float(clean_passthrough_prob)
    self.min_n_apply = int(min_n_apply)
    self.max_n_apply = int(max_n_apply)
    # If start values left at default (-1), default them to the final values
    # (equivalent to "no ramp"). This keeps backward compat with old gin.
    self.min_n_apply_start = (int(min_n_apply_start)
                              if min_n_apply_start >= 0
                              else self.min_n_apply)
    self.max_n_apply_start = (int(max_n_apply_start)
                              if max_n_apply_start >= 0
                              else self.max_n_apply)
    self._n_apply_schedule_steps = int(n_apply_schedule_steps)
    self.preserve_loudness = bool(preserve_loudness)
    self.max_loudness_gain_db = float(max_loudness_gain_db)
    self._wet_schedule_steps = int(wet_schedule_steps)

    # ----- Schedule step counter (NOT checkpointed) -----
    # We bypass Keras auto-tracking via object.__setattr__, otherwise this
    # tf.Variable would be saved into the checkpoint and silently overwrite
    # the constructor's `wet_schedule_start_step` on resume — exactly the
    # bug that bit basswave_v2 (sched_step was restored to 0 from v1's
    # ckpt, ignoring the user's wet_schedule_start_step=76000).
    #
    # Trade-off: if you save+restore mid-schedule across runs, the counter
    # restarts at `wet_schedule_start_step` rather than at where it was.
    # That's intentional — we want the wet schedule to be a session-local
    # property, fully controlled by gin params, not a hidden ckpt state.
    object.__setattr__(self, '_sched_step', tf.Variable(
        int(wet_schedule_start_step), trainable=False,
        dtype=tf.int64, name='degradation_sched_step'))

    # Store initial wet values (0..1) for the schedule computation.
    self._wet_starts = [
        float(phasing_wet) / 100.0,
        float(ghosting_wet) / 100.0,
        float(wrong_eq_wet) / 100.0,
        float(compression_wet) / 100.0,
        float(dynamic_eq_wet) / 100.0,
        float(bandwidth_wet) / 100.0,
    ]
    if not (1 <= self.min_n_apply <= self.max_n_apply <= 6):
      raise ValueError(
          'Need 1 <= min_n_apply <= max_n_apply <= 6, '
          f'got {min_n_apply} / {max_n_apply}.')
    if not (1 <= self.min_n_apply_start <= self.max_n_apply_start <= 6):
      raise ValueError(
          'Need 1 <= min_n_apply_start <= max_n_apply_start <= 6, '
          f'got {self.min_n_apply_start} / {self.max_n_apply_start}.')

    # Build the 6 layer instances. Order = canonical pipeline order.
    self.layers_in_order = [
        Phasing(wet_dry_pct=phasing_wet, sample_rate=sample_rate),
        Ghosting(wet_dry_pct=ghosting_wet, sample_rate=sample_rate),
        WrongEQ(wet_dry_pct=wrong_eq_wet, sample_rate=sample_rate),
        AggressiveCompression(wet_dry_pct=compression_wet,
                              sample_rate=sample_rate),
        DynamicEQ(wet_dry_pct=dynamic_eq_wet, sample_rate=sample_rate),
        BandwidthLimit(wet_dry_pct=bandwidth_wet, sample_rate=sample_rate),
    ]
    self.n_total = len(self.layers_in_order)

  def _schedule_t(self, schedule_steps):
    """Return ramp progress in [0, 1] for the given schedule length."""
    if schedule_steps <= 0:
      return tf.constant(1.0)
    return tf.minimum(
        tf.cast(self._sched_step, tf.float32) / float(schedule_steps),
        1.0)

  def _update_wet_values(self):
    """Linearly ramp each layer's _wet from its start value to 1.0.

    No-op when wet_schedule_steps == 0 (static wet/dry, legacy behaviour).
    """
    if self._wet_schedule_steps <= 0:
      return
    t = self._schedule_t(self._wet_schedule_steps)
    for layer, w0 in zip(self.layers_in_order, self._wet_starts):
      layer._wet.assign(w0 + (1.0 - w0) * t)

  def _draw_selection_mask(self):
    """Draw [n_total] mask: 1 = layer selected this batch, 0 = skip.

    K ~ uniform[cur_min_n_apply, cur_max_n_apply] (inclusive), where
    cur_min/cur_max are linearly ramped from start to final values over
    n_apply_schedule_steps.
    """
    if self._n_apply_schedule_steps > 0:
      t = self._schedule_t(self._n_apply_schedule_steps)
      cur_min = tf.cast(tf.round(
          float(self.min_n_apply_start) +
          float(self.min_n_apply - self.min_n_apply_start) * t), tf.int32)
      cur_max = tf.cast(tf.round(
          float(self.max_n_apply_start) +
          float(self.max_n_apply - self.max_n_apply_start) * t), tf.int32)
      # Guard: ensure cur_min <= cur_max even with rounding noise.
      cur_max = tf.maximum(cur_max, cur_min)
    else:
      cur_min = tf.constant(self.min_n_apply, dtype=tf.int32)
      cur_max = tf.constant(self.max_n_apply, dtype=tf.int32)

    n_apply = tf.random.uniform([], cur_min, cur_max + 1, dtype=tf.int32)
    keys = tf.random.uniform([self.n_total])
    rank = tf.argsort(tf.argsort(keys))
    return tf.cast(rank < n_apply, tf.float32)                 # [n_total]

  def _normalize_loudness(self, audio_in, audio_out):
    """Rescale audio_out so per-example RMS matches audio_in."""
    rms_in = tf.sqrt(
        tf.reduce_mean(audio_in * audio_in, axis=-1, keepdims=True) + 1e-10)
    rms_out = tf.sqrt(
        tf.reduce_mean(audio_out * audio_out, axis=-1, keepdims=True) + 1e-10)
    gain = rms_in / rms_out
    max_gain_lin = tf.pow(10.0, self.max_loudness_gain_db / 20.0)
    gain = tf.minimum(gain, max_gain_lin)
    return audio_out * gain

  def call(self, audio_clean, training=True):
    """Apply random subset of degradations.

    Args:
      audio_clean: [batch, n_samples] mono float audio in approximately [-1, 1].
      training: if False, returns audio_clean unchanged.
    """
    if not training:
      return audio_clean

    # Update per-layer wet values (no-op if wet_schedule_steps == 0).
    self._update_wet_values()

    # Draw scheduled K-mask.
    sel_mask = self._draw_selection_mask()                     # [6]

    # Increment counter ONCE per call (drives BOTH wet ramp and n_apply ramp).
    self._sched_step.assign_add(1)

    audio = audio_clean
    for i, layer in enumerate(self.layers_in_order):
      m = sel_mask[i]
      degraded = layer(audio, training=True)
      audio = m * degraded + (1.0 - m) * audio

    if self.preserve_loudness:
      audio = self._normalize_loudness(audio_clean, audio)

    # Per-example clean passthrough.
    batch = tf.shape(audio_clean)[0]
    is_clean = tf.cast(
        tf.random.uniform([batch, 1]) < self.clean_passthrough_prob,
        tf.float32)
    audio = is_clean * audio_clean + (1.0 - is_clean) * audio
    return audio
