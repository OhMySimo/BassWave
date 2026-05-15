# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Mono phasing degradation.

Mathematical model: y(t) = 0.5 * x(t) + 0.5 * x(t - tau)

where tau is a random delay, drawn per example, in [-max_delay_ms, +max_delay_ms].
Negative tau means the "delayed" copy actually leads the original (achieved by
shifting in the opposite direction).

This produces comb filtering: the magnitude response has notches at
f_n = (2n+1) / (2 * |tau|), n = 0, 1, 2, ...

For tau = 1 ms: first notch at 500 Hz.
For tau = 20 ms: first notch at 25 Hz.

The full ±20 ms range covers everything from "subtle smearing" (sub-millisecond)
to "very obvious phasing" (tens of milliseconds, beyond the precedence-effect
window where the ear stops fusing the two arrivals).
"""

import tensorflow.compat.v2 as tf

from ddsp.training.degradations.base import BaseDegradation


class Phasing(BaseDegradation):
  """Mono phaser via single-tap random delay + 50/50 mix."""

  def __init__(self,
               wet_dry_pct: float = 50.0,
               sample_rate: int = 44100,
               max_delay_ms: float = 20.0,
               name: str = 'phasing',
               **kwargs):
    super().__init__(wet_dry_pct=wet_dry_pct, sample_rate=sample_rate,
                     name=name, **kwargs)
    self.max_delay_samples = int(max_delay_ms * 1e-3 * sample_rate)

  def _degrade(self, audio):
    # audio: [B, T]
    batch = tf.shape(audio)[0]
    n = tf.shape(audio)[1]
    pad = self.max_delay_samples

    # Pad so we can index in [-pad, +pad] without going OOB.
    padded = tf.pad(audio, [[0, 0], [pad, pad]])  # [B, T + 2*pad]

    # Per-example integer delay in [-pad, +pad].
    # Note: in principle a fractional delay (with linear-interp) would give
    # slightly nicer notch placement, but at 44.1 kHz one sample = 22.7 us,
    # well below auditory resolution for the comb-filter peak frequencies
    # we care about. Integer delay is cheap and graph-friendly.
    delays = tf.random.uniform([batch], -pad, pad + 1, dtype=tf.int32)

    # Build [B, T] gather index. For example b: idx[b, t] = pad + t + delays[b]
    base_idx = tf.range(n)[tf.newaxis, :]                       # [1, T]
    offsets = (pad + delays)[:, tf.newaxis]                     # [B, 1]
    idx = base_idx + offsets                                    # [B, T]
    delayed = tf.gather(padded, idx, batch_dims=1)              # [B, T]

    return 0.5 * audio + 0.5 * delayed
