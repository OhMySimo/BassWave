# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Aggressive compression / limiting / clipping degradation.

Models the kind of damage done by an overzealous mastering pass:
  * Heavy downward compression (low threshold + high ratio)
  * Hard limiter pushing everything to the ceiling
  * Inter-sample clipping or driving into a soft-clip stage

Pipeline applied in this order (each can be on or off per example):
  1. RMS-window envelope, gain reduction above threshold (compressor)
  2. Make-up gain pushes the signal back toward 0 dBFS
  3. Soft clip via tanh (saturation / character distortion)
  4. Hard ceiling (limiter — final brickwall)

Implementation notes:
  * Envelope detector: 5 ms RMS window via 1D convolution. Faster than a
    proper IIR detector and graph-friendly. Tradeoff: no separate
    attack/release time constants — the envelope smooths symmetrically.
    Acceptable for augmentation; if it sounds too synthetic in practice,
    swap for `tf.scan`-based one-pole at the cost of speed.
  * Hard clip + soft clip together produce realistic mastering-chain damage.
"""

import tensorflow.compat.v2 as tf

from ddsp.training.degradations.base import (
    BaseDegradation, db_to_lin, power_to_db)


class AggressiveCompression(BaseDegradation):
  """Compressor + limiter + soft clip + hard clip cascade."""

  def __init__(self,
               wet_dry_pct: float = 50.0,
               sample_rate: int = 44100,
               threshold_db_min: float = -25.0,
               threshold_db_max: float = -10.0,
               ratio_min: float = 4.0,
               ratio_max: float = 20.0,
               makeup_db_min: float = 3.0,
               makeup_db_max: float = 12.0,
               soft_clip_k_min: float = 1.0,
               soft_clip_k_max: float = 4.0,
               hard_ceiling: float = 0.99,
               rms_window_ms: float = 5.0,
               preserve_loudness: bool = True,
               name: str = 'aggressive_compression',
               **kwargs):
    super().__init__(wet_dry_pct=wet_dry_pct, sample_rate=sample_rate,
                     name=name, **kwargs)
    self.threshold_db_range = (threshold_db_min, threshold_db_max)
    self.ratio_range = (ratio_min, ratio_max)
    self.makeup_db_range = (makeup_db_min, makeup_db_max)
    self.soft_clip_k_range = (soft_clip_k_min, soft_clip_k_max)
    self.hard_ceiling = float(hard_ceiling)
    self.rms_window = max(1, int(rms_window_ms * 1e-3 * sample_rate))
    self.preserve_loudness = bool(preserve_loudness)

  def _envelope_db(self, audio):
    """RMS-window envelope in dB. [B, T] → [B, T]."""
    sq = audio * audio
    kernel = tf.ones([self.rms_window, 1, 1], dtype=tf.float32) / float(
        self.rms_window)
    sq_4d = sq[:, :, tf.newaxis]
    rms_sq = tf.nn.conv1d(sq_4d, kernel, stride=1, padding='SAME')
    rms_sq = tf.squeeze(rms_sq, axis=-1)
    return power_to_db(rms_sq)

  def _degrade(self, audio):
    batch = tf.shape(audio)[0]

    # Per-example random params.
    threshold_db = tf.random.uniform(
        [batch, 1], self.threshold_db_range[0], self.threshold_db_range[1])
    ratio = tf.random.uniform(
        [batch, 1], self.ratio_range[0], self.ratio_range[1])
    makeup_db = tf.random.uniform(
        [batch, 1], self.makeup_db_range[0], self.makeup_db_range[1])
    clip_k = tf.random.uniform(
        [batch, 1], self.soft_clip_k_range[0], self.soft_clip_k_range[1])

    # 1. Compute envelope.
    env_db = self._envelope_db(audio)                          # [B, T]

    # 2. Gain reduction: for env_db > threshold, attenuate by
    #    (env_db - threshold) * (1 - 1/ratio).
    over = tf.maximum(env_db - threshold_db, 0.0)
    gain_db = -over * (1.0 - 1.0 / ratio)
    gain_lin = db_to_lin(gain_db)                              # [B, T]

    # 3. Apply gain + makeup gain.
    compressed = audio * gain_lin * db_to_lin(makeup_db)

    # 4. Soft clip (tanh-based saturation).
    soft = tf.tanh(clip_k * compressed) / (tf.tanh(clip_k) + 1e-8)

    # 5. Hard ceiling — final brickwall.
    out = tf.clip_by_value(soft, -self.hard_ceiling, self.hard_ceiling)

    # 6. Loudness preservation: rescale to match input RMS per-example.
    # The makeup_gain pumps the signal toward 0 dBFS, which would let the
    # downstream model "detect compression by loudness" instead of by the
    # actual compression artefacts (dynamic range reduction, soft-clip
    # distortion). We match RMS to remove that information leak while
    # keeping the artefacts intact.
    if self.preserve_loudness:
      rms_in = tf.sqrt(
          tf.reduce_mean(audio * audio, axis=-1, keepdims=True) + 1e-10)
      rms_out = tf.sqrt(
          tf.reduce_mean(out * out, axis=-1, keepdims=True) + 1e-10)
      gain = rms_in / rms_out
      # Cap at 12 dB to avoid amplifying near-silent residuals.
      gain = tf.minimum(gain, tf.constant(3.98, dtype=tf.float32))  # 12 dB
      out = out * gain

    return out
