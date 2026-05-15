# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Base class for BassWave on-the-fly audio degradation layers.

Every degradation is a `tf.keras.layers.Layer` that:
  * Takes mono audio shaped [batch, n_samples], values approximately in [-1, 1].
  * Has a `wet_dry_pct` parameter in [0, 100] (audio-plugin semantics).
  * Produces a degraded version internally with per-example randomised params.
  * Returns linear blend: out = (1 - w) * dry + w * wet.

Subclasses override `_degrade(audio)` to produce the fully-wet output. The
public `call()` handles the wet/dry mix and any `training=False` short-circuit.

Per-example randomness: every random draw inside `_degrade` should include
the batch dimension, so different examples in the same batch get different
parameter values. Per-batch randomness (single value drawn once per call,
shared across the batch) is also fine for parameters where per-example
diversity isn't critical.
"""

import tensorflow.compat.v2 as tf


class BaseDegradation(tf.keras.layers.Layer):
  """Abstract base class. Handles wet/dry blending; subclasses do the work."""

  def __init__(self,
               wet_dry_pct: float = 50.0,
               sample_rate: int = 44100,
               name: str = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.sample_rate = int(sample_rate)
    # tf.Variable so that DegradationPipeline can update _wet at runtime
    # (e.g. for a linear wet schedule over training steps).
    self._wet = tf.Variable(
        max(0.0, min(100.0, float(wet_dry_pct))) / 100.0,
        trainable=False, dtype=tf.float32, name=f'{name}_wet')

  @property
  def is_active(self) -> bool:
    """True if wet > 0 (otherwise the layer is a no-op)."""
    return float(self._wet.numpy()) > 1e-6 if tf.executing_eagerly() else True

  def _degrade(self, audio):
    """Subclasses override this. Must return tensor of same shape as input."""
    raise NotImplementedError

  def call(self, audio, training=True):
    if not training:
      return audio
    # Short-circuit when wet=0 to save compute.
    if not self.is_active:
      return audio
    wet = self._degrade(audio)
    return (1.0 - self._wet) * audio + self._wet * wet


# -----------------------------------------------------------------------------
# Helpers shared across degradation modules.
# -----------------------------------------------------------------------------


def db_to_lin(db):
  """10^(dB/20). Works on scalar or tensor."""
  return tf.pow(10.0, db / 20.0)


def lin_to_db(lin, eps=1e-12):
  """20 * log10(|lin| + eps)."""
  return 20.0 * tf.math.log(tf.abs(lin) + eps) / tf.math.log(10.0)


def power_to_db(power, eps=1e-12):
  """10 * log10(power + eps)."""
  return 10.0 * tf.math.log(power + eps) / tf.math.log(10.0)


def fft_freqs(n_freq, sample_rate):
  """Return [n_freq] tensor of bin centre frequencies in Hz, 0..Nyquist."""
  return tf.linspace(0.0, sample_rate / 2.0, n_freq)
