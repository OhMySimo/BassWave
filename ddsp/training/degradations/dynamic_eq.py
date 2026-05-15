# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Dynamic EQ degradation — frequency-band gain that reacts to band level.

Models the case where a dynamic EQ on a mix bus tames a specific band of
the bass when it gets too loud — e.g. a peaking band at 100 Hz with Q=2 that
attenuates -3 dB whenever the energy at that band exceeds a threshold.

Per-example random parameters (NOW configurable via gin):
  * Band centre frequency f_c    : [fc_min, fc_max]      (default 80-500 Hz)
  * Band width (octaves)          : [bw_octaves_min, bw_octaves_max]
                                    (default 0.3 - 1.0 octaves, Q ~ 1.4 - 5)
  * Threshold offset below max    : [threshold_offset_db_min,
                                     threshold_offset_db_max]
                                    (default 5 - 25 dB; lower = triggers more
                                    often)
  * Maximum gain reduction        : [max_reduction_db_min,
                                     max_reduction_db_max]
                                    (default -8 to -2 dB; more negative =
                                    deeper cut)

For the strong-ramp curriculum we recommend boosting:
  * threshold_offset_db_min/max → 3 / 15  (triggers more often)
  * max_reduction_db_min/max → -15 / -5  (deeper cut)

Implementation: STFT, identify the target band via a Gaussian-in-log-frequency
mask, compute per-frame band energy, derive per-frame gain reduction
proportional to how much the band energy exceeds threshold, apply the
reduction only to bins inside the band, iSTFT back.
"""

import tensorflow.compat.v2 as tf

from ddsp.training.degradations.base import (
    BaseDegradation, db_to_lin, power_to_db, fft_freqs)


class DynamicEQ(BaseDegradation):
  """Time-varying band attenuation, threshold-driven."""

  def __init__(self,
               wet_dry_pct: float = 50.0,
               sample_rate: int = 44100,
               n_fft: int = 2048,
               hop_length: int = 512,
               # Band centre frequency range (Hz).
               fc_min: float = 80.0,
               fc_max: float = 500.0,
               # Band width range, in octaves (FWHM).
               bw_octaves_min: float = 0.3,
               bw_octaves_max: float = 1.0,
               # Threshold offset below per-clip peak (dB; >0 = below peak).
               threshold_offset_db_min: float = 5.0,
               threshold_offset_db_max: float = 25.0,
               # Maximum gain reduction at peak (dB; negative = cut).
               max_reduction_db_min: float = -8.0,
               max_reduction_db_max: float = -2.0,
               name: str = 'dynamic_eq',
               **kwargs):
    super().__init__(wet_dry_pct=wet_dry_pct, sample_rate=sample_rate,
                     name=name, **kwargs)
    self.n_fft = int(n_fft)
    self.hop_length = int(hop_length)
    self.fc_range = (float(fc_min), float(fc_max))
    self.bw_octaves_range = (float(bw_octaves_min), float(bw_octaves_max))
    self.threshold_offset_db_range = (
        float(threshold_offset_db_min), float(threshold_offset_db_max))
    self.max_reduction_db_range = (
        float(max_reduction_db_min), float(max_reduction_db_max))

    if self.fc_range[0] >= self.fc_range[1]:
      raise ValueError(
          f'fc_min ({fc_min}) must be < fc_max ({fc_max}).')
    if self.max_reduction_db_range[0] >= self.max_reduction_db_range[1]:
      raise ValueError(
          'max_reduction_db_min must be < max_reduction_db_max, got '
          f'{max_reduction_db_min} / {max_reduction_db_max}.')

  def _degrade(self, audio):
    batch = tf.shape(audio)[0]
    n = tf.shape(audio)[1]

    # --- Per-example random params (configurable ranges) ---
    fc = tf.random.uniform([batch], *self.fc_range)
    bw_octaves = tf.random.uniform([batch], *self.bw_octaves_range)
    threshold_offset_db = tf.random.uniform(
        [batch, 1], *self.threshold_offset_db_range)
    max_reduction_db = tf.random.uniform(
        [batch, 1], *self.max_reduction_db_range)

    # --- STFT ---
    spec = tf.signal.stft(
        audio, frame_length=self.n_fft, frame_step=self.hop_length,
        pad_end=True)                                          # [B, T_f, F]
    n_freq = tf.shape(spec)[-1]
    freqs = fft_freqs(n_freq, self.sample_rate)                # [F]

    # --- Per-example Gaussian-in-log-frequency band mask ---
    sigma = bw_octaves / 2.355
    log2 = tf.math.log(2.0)
    log_freqs = tf.math.log(freqs + 1e-3) / log2
    log_fc = tf.math.log(fc + 1e-3) / log2
    log_ratio = log_freqs[tf.newaxis, :] - log_fc[:, tf.newaxis]
    band_mask = tf.exp(
        -0.5 * (log_ratio / sigma[:, tf.newaxis]) ** 2)

    # --- Band energy per frame ---
    spec_mag2 = tf.cast(tf.abs(spec) ** 2, tf.float32)
    band_energy = tf.reduce_sum(
        spec_mag2 * band_mask[:, tf.newaxis, :], axis=-1)
    band_db = power_to_db(band_energy)

    # --- Threshold relative to per-example max band energy ---
    max_db = tf.reduce_max(band_db, axis=-1, keepdims=True)
    threshold_db = max_db - threshold_offset_db

    # --- Gain reduction per frame ---
    over = tf.maximum(band_db - threshold_db, 0.0)
    over_max = tf.reduce_max(over, axis=-1, keepdims=True) + 1e-6
    gain_db_frame = max_reduction_db * (over / over_max)
    gain_lin_frame = db_to_lin(gain_db_frame)

    # --- Per-bin gain ---
    gain_per_bin = 1.0 + (gain_lin_frame[:, :, tf.newaxis] - 1.0) * \
        band_mask[:, tf.newaxis, :]

    spec_out = spec * tf.cast(gain_per_bin, spec.dtype)

    audio_out = tf.signal.inverse_stft(
        spec_out,
        frame_length=self.n_fft,
        frame_step=self.hop_length,
        window_fn=tf.signal.inverse_stft_window_fn(self.hop_length))
    audio_out = audio_out[:, :n]
    pad_needed = n - tf.shape(audio_out)[-1]
    audio_out = tf.cond(
        pad_needed > 0,
        lambda: tf.pad(audio_out, [[0, 0], [0, pad_needed]]),
        lambda: audio_out)
    return audio_out
