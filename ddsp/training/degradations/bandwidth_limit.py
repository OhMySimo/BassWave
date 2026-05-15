# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Random bandwidth-limit degradation.

Simulates the effect of downsampling to a random low rate (500 Hz to 44.1 kHz)
and resampling back to 44.1 kHz. The audible artifact is bandwidth limitation
(loss of HF content above target Nyquist) — exactly what real lossy codec
chains and accidental resampling pipelines produce.

The user's original spec was:
  > downsampling randomico da 44.1khz fino a sample rate 500hz, con
  > riconversione in 44.1khz prima di entrare nel loop di training

We don't actually change tensor sizes (which would break batch shapes inside
a tf.function). Instead we approximate the same effect via STFT mask:
zero out frequency bins above target_sr / 2.

Why this is equivalent in audible content:
  * Ideal downsample-then-upsample = bandlimit + decimate + zero-pad + interpolate
  * Decimate + zero-pad + ideal interpolation = identity (Shannon)
  * So the ONLY destructive step is the bandlimit. Doing just the bandlimit
    in-place at 44.1 k achieves the same auditory result without resizing.

Per-example: target_sr drawn log-uniform in [500, 44100]. Log-uniform so
extreme bandlimits (sub-1 kHz) are not vanishingly rare in training.
"""

import tensorflow.compat.v2 as tf

from ddsp.training.degradations.base import (
    BaseDegradation, fft_freqs)


class BandwidthLimit(BaseDegradation):
  """Random STFT-mask bandlimit simulating a downsample-then-upsample chain."""

  def __init__(self,
               wet_dry_pct: float = 50.0,
               sample_rate: int = 44100,
               min_target_sr: int = 500,
               max_target_sr: int = 44100,
               transition_width_hz: float = 50.0,
               n_fft: int = 2048,
               hop_length: int = 512,
               name: str = 'bandwidth_limit',
               **kwargs):
    super().__init__(wet_dry_pct=wet_dry_pct, sample_rate=sample_rate,
                     name=name, **kwargs)
    self.min_target_sr = int(min_target_sr)
    self.max_target_sr = int(max_target_sr)
    self.transition_width_hz = float(transition_width_hz)
    self.n_fft = int(n_fft)
    self.hop_length = int(hop_length)

  def _degrade(self, audio):
    batch = tf.shape(audio)[0]
    n = tf.shape(audio)[1]

    # --- Per-example log-uniform target SR → cutoff = SR/2 ---
    log_low = tf.math.log(float(self.min_target_sr))
    log_high = tf.math.log(float(self.max_target_sr))
    target_sr = tf.exp(tf.random.uniform([batch], log_low, log_high))
    cutoff = target_sr / 2.0                                   # [B]

    # --- STFT, build smooth-transition lowpass mask, multiply, iSTFT ---
    spec = tf.signal.stft(
        audio, frame_length=self.n_fft, frame_step=self.hop_length,
        pad_end=True)                                          # [B, T_f, F]
    n_freq = tf.shape(spec)[-1]
    freqs = fft_freqs(n_freq, self.sample_rate)                # [F]

    # Sigmoid roll-off centred on cutoff: 1 below cutoff, 0 above.
    # transition_width sets how steep the roll-off is.
    mask = tf.sigmoid(
        -(freqs[tf.newaxis, :] - cutoff[:, tf.newaxis]) /
        self.transition_width_hz)                              # [B, F]

    spec_filt = spec * tf.cast(mask[:, tf.newaxis, :], spec.dtype)

    audio_out = tf.signal.inverse_stft(
        spec_filt,
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
