# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Wrong-EQ degradation — extreme EQ pathologies applied in frequency domain.

Five pathology types, randomly chosen per example:
  0. bass_boost   — peaking +8..+15 dB at 40-80 Hz, normal Q (0.7-1.5)   [ADDITIVE]
  1. low_cut      — high-pass at 80-200 Hz, 2nd-order Butterworth         [DESTRUCTIVE]
  2. high_boost   — high-shelf +8..+15 dB above 3-8 kHz                   [ADDITIVE]
  3. mid_scoop    — peaking -8..-12 dB at 800-2000 Hz, normal Q (0.5-1.0) [DESTRUCTIVE]
  4. resonance    — narrow peak +10..+18 dB at 200-4000 Hz, very high Q   [ADDITIVE]

Implementation: full-buffer rfft, multiply spectrum by per-example gain
curve, irfft back. Per-example dispatch via einsum over a stacked
[5, batch, n_freq] tensor of pre-computed curves.

`destructive_bias` (NEW, 0..1):
  Probability that an example draws from the destructive subset {low_cut,
  mid_scoop} rather than the full uniform 5-way distribution. The other
  three types only ADD energy (boosts / resonance) — they don't remove
  information the model has to recover. For a denoising AE, biasing toward
  destructive types makes the reconstruction problem harder, which is what
  you want.
    0.0 = legacy uniform (each of 5 types with p=0.2)
    0.5 = roughly half of examples destructive
    1.0 = always destructive (only low_cut / mid_scoop)
  Recommended for the strong-ramp curriculum: 0.7.
"""

import tensorflow.compat.v2 as tf

from ddsp.training.degradations.base import (
    BaseDegradation, db_to_lin, fft_freqs)


# -----------------------------------------------------------------------------
# Filter shape primitives. All take freqs:[F] + per-example params:[B], return
# magnitude curves: [B, F].
# -----------------------------------------------------------------------------


def _peak_curve(freqs, fc, q, gain_db):
  """Lorentzian peak in linear frequency.

  H(f) = 1 + (A - 1) / (1 + ((f - fc) / bw)^2)
  """
  fc = fc[:, tf.newaxis]
  q = q[:, tf.newaxis]
  A = db_to_lin(gain_db)[:, tf.newaxis]
  bw = fc / tf.maximum(q, 1e-3)
  f = freqs[tf.newaxis, :]
  return 1.0 + (A - 1.0) / (1.0 + ((f - fc) / bw) ** 2)


def _highpass_curve(freqs, fc, order=2):
  """Butterworth-shape high-pass magnitude.

  |H(f)| = (f/fc)^n / sqrt(1 + (f/fc)^(2n))
  """
  fc = fc[:, tf.newaxis]
  f = freqs[tf.newaxis, :]
  ratio = tf.maximum(f / tf.maximum(fc, 1e-3), 1e-6)
  num = tf.pow(ratio, float(order))
  den = tf.sqrt(1.0 + tf.pow(ratio, 2.0 * float(order)))
  return num / den


def _highshelf_curve(freqs, fc, gain_db):
  """High-shelf magnitude curve (sigmoid-shaped transition in log freq)."""
  fc = fc[:, tf.newaxis]
  A = db_to_lin(gain_db)[:, tf.newaxis]
  f = freqs[tf.newaxis, :]
  log_ratio = tf.math.log(tf.maximum(f, 1e-3) / tf.maximum(fc, 1e-3))
  weight = tf.sigmoid(log_ratio)
  return 1.0 + (A - 1.0) * weight


# -----------------------------------------------------------------------------
# Layer.
# -----------------------------------------------------------------------------


# Indices into the 5-curve stack:
_BASS_BOOST = 0       # additive
_LOW_CUT = 1          # destructive
_HIGH_BOOST = 2       # additive
_MID_SCOOP = 3        # destructive
_RESONANCE = 4        # additive

_DESTRUCTIVE = (_LOW_CUT, _MID_SCOOP)
_ADDITIVE = (_BASS_BOOST, _HIGH_BOOST, _RESONANCE)


class WrongEQ(BaseDegradation):
  """Apply a randomly-chosen extreme EQ pathology, in frequency domain."""

  N_TYPES = 5

  def __init__(self,
               wet_dry_pct: float = 50.0,
               sample_rate: int = 44100,
               destructive_bias: float = 0.0,
               name: str = 'wrong_eq',
               **kwargs):
    super().__init__(wet_dry_pct=wet_dry_pct, sample_rate=sample_rate,
                     name=name, **kwargs)
    if not (0.0 <= float(destructive_bias) <= 1.0):
      raise ValueError(
          f'destructive_bias must be in [0, 1], got {destructive_bias}')
    self.destructive_bias = float(destructive_bias)

  def _draw_eq_choice(self, batch):
    """Per-example index in [0, 5) with destructive_bias-weighted choice.

    Implementation: pick a destructive index (uniform in {1, 3}) and an
    additive index (uniform in {0, 2, 4}) per example, then blend with a
    Bernoulli mask of probability `destructive_bias`.
    """
    is_destructive = tf.random.uniform([batch]) < self.destructive_bias

    # Destructive choice: idx in {0, 1} → maps to {1, 3}.
    d_pick = tf.random.uniform([batch], 0, 2, dtype=tf.int32)
    dest_choice = tf.where(d_pick < 1,
                           tf.fill([batch], _LOW_CUT),
                           tf.fill([batch], _MID_SCOOP))

    # Additive choice: idx in {0, 1, 2} → maps to {0, 2, 4}.
    a_pick = tf.random.uniform([batch], 0, 3, dtype=tf.int32)
    add_choice = tf.where(a_pick < 1,
                          tf.fill([batch], _BASS_BOOST),
                          tf.where(a_pick < 2,
                                   tf.fill([batch], _HIGH_BOOST),
                                   tf.fill([batch], _RESONANCE)))

    return tf.where(is_destructive, dest_choice, add_choice)

  def _degrade(self, audio):
    batch = tf.shape(audio)[0]
    n = tf.shape(audio)[1]

    spec = tf.signal.rfft(audio)                              # [B, n//2+1]
    n_freq = tf.shape(spec)[-1]
    freqs = fft_freqs(n_freq, self.sample_rate)               # [F]

    # Per-example random params for ALL 5 patterns.
    bb = _peak_curve(
        freqs,
        fc=tf.random.uniform([batch], 40.0, 80.0),
        q=tf.random.uniform([batch], 0.7, 1.5),
        gain_db=tf.random.uniform([batch], 8.0, 15.0))
    lc = _highpass_curve(
        freqs,
        fc=tf.random.uniform([batch], 80.0, 200.0),
        order=2)
    hb = _highshelf_curve(
        freqs,
        fc=tf.random.uniform([batch], 3000.0, 8000.0),
        gain_db=tf.random.uniform([batch], 8.0, 15.0))
    ms = _peak_curve(
        freqs,
        fc=tf.random.uniform([batch], 800.0, 2000.0),
        q=tf.random.uniform([batch], 0.5, 1.0),
        gain_db=tf.random.uniform([batch], -12.0, -8.0))
    rs = _peak_curve(
        freqs,
        fc=tf.random.uniform([batch], 200.0, 4000.0),
        q=tf.random.uniform([batch], 5.0, 15.0),
        gain_db=tf.random.uniform([batch], 10.0, 18.0))

    all_curves = tf.stack([bb, lc, hb, ms, rs], axis=0)        # [5, B, F]

    # Per-example dispatch with destructive_bias.
    eq_choice = self._draw_eq_choice(batch)                    # [B]
    one_hot = tf.one_hot(eq_choice, self.N_TYPES,
                         dtype=tf.float32)                     # [B, 5]
    curve = tf.einsum('bk,kbf->bf', one_hot, all_curves)       # [B, F]

    spec_eq = spec * tf.cast(curve, spec.dtype)
    audio_out = tf.signal.irfft(spec_eq, fft_length=[n])
    return audio_out
