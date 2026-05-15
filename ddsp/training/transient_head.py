# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""TransientHead — per-frame onset / offset / silence prediction head.

Takes `z` [B, T, z_dim] from the encoder and predicts 6 per-frame quantities:
  1. onset_logit       [B, T, 1]  — BCE target: onset_mask
  2. onset_offset_norm [B, T, 1]  — MSE target: onset_offset_samples /
                                    half_frame_samples  ∈ [-1, +1]
  3. onset_velocity    [B, T, 1]  — MSE target: onset_velocity ∈ [0, 1]
  4. offset_logit      [B, T, 1]  — BCE target: offset_mask
  5. offset_offset_norm[B, T, 1]  — MSE target: offset_offset_samples /
                                    half_frame_samples  ∈ [-1, +1]
  6. silence_logit     [B, T, 1]  — BCE target: silence_mask

Architecture: 3 × Conv1D (causal optional, default non-causal) on z, then
a linear projection. Causal variant is useful if the head is ever used for
real-time streaming inference; non-causal gives ±1 frame of context which is
enough to improve onset localisation.

Loss (TransientLoss)
--------------------
  L = λ_on  * focal_bce(onset_logit,    onset_mask,   pos_weight)
    + λ_sub  * masked_mse(onset_offset_norm, onset_mask)
    + λ_vel  * masked_mse(onset_velocity,    onset_mask)
    + λ_off  * focal_bce(offset_logit,   offset_mask,  pos_weight)
    + λ_sil  * bce(silence_logit,        silence_mask, sil_pos_weight)

BCE uses a *weighted* form: positive class gets `pos_weight` times more
gradient. Default 10.0 for onset/offset (rare class), 2.0 for silence.

"Focal" modulation (gamma > 0) further down-weights easy negatives:
  FL = -(1 - p_t)^gamma * log(p_t)
With gamma=0, this reduces to standard BCE. Default gamma=1.5 for onset.

Masked MSE: only computed where the binary mask==1. If all mask==0 in a
batch (can happen with aggressive clean_passthrough), loss is 0.

Normalisation note: onset_offset_samples and offset_offset_samples stored
in TFRecord are in raw samples ∈ [-441, +441] at 50 Hz / 44.1 kHz. We
normalise by half_frame_samples = sample_rate / (2 * frame_rate) =
44100 / 100 = 441 before computing MSE, so the loss is always in [-1, +1]
regardless of sample rate.

Inference usage
---------------
At inference (preset mode), the only feature you need is `silence_logit`:

    silence_prob = tf.sigmoid(outputs['silence_logit'])   # [B, T, 1]
    amps = amps * (1.0 - silence_prob)

This gates the harmonic synthesiser amplitude to 0 during predicted silent
frames, fixing the "basso suona come MIDI che non rispetta le pause" problem.

The `onset_logit` can optionally be used to inject a brief loudness transient
at predicted onset positions (not implemented here; see basswave_infer.py).
"""
from __future__ import annotations

from typing import Optional

import gin
import tensorflow.compat.v2 as tf


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _weighted_bce(logits: tf.Tensor,
                  targets: tf.Tensor,
                  pos_weight: float,
                  gamma: float = 0.0) -> tf.Tensor:
  """Per-element weighted BCE with optional focal modulation.

  Args:
    logits: [B, T, 1] float32, raw (un-sigmoided) predictions.
    targets: [B, T, 1] float32 in {0, 1}.
    pos_weight: positive class weight (>1 = upweight positives).
    gamma: focal exponent. 0 = standard BCE.

  Returns:
    Scalar mean loss.
  """
  # Standard TF weighted BCE: weight the positive class by pos_weight.
  # tf.nn.weighted_cross_entropy_with_logits signature:
  #   loss = targets * -log(sigmoid(logits)) * pos_weight
  #        + (1-targets) * -log(1-sigmoid(logits))
  pw = tf.constant(pos_weight, dtype=tf.float32)
  per_elem = tf.nn.weighted_cross_entropy_with_logits(
      labels=targets, logits=logits, pos_weight=pw)

  if gamma > 0.0:
    # Focal weight: (1 - p_t)^gamma where p_t = sigmoid if target==1
    #                                          else 1-sigmoid if target==0
    p = tf.sigmoid(logits)
    p_t = targets * p + (1.0 - targets) * (1.0 - p)
    focal_w = tf.pow(1.0 - p_t, gamma)
    per_elem = focal_w * per_elem

  return tf.reduce_mean(per_elem)


def _masked_mse(predictions: tf.Tensor,
                targets: tf.Tensor,
                mask: tf.Tensor) -> tf.Tensor:
  """MSE computed only where mask == 1.

  Returns 0 when mask is all-zero (no onset in batch).

  Args:
    predictions: [B, T, 1] float32.
    targets:     [B, T, 1] float32.
    mask:        [B, T, 1] float32 in {0, 1}.

  Returns:
    Scalar loss.
  """
  n = tf.reduce_sum(mask)
  sq_err = tf.square(predictions - targets) * mask
  return tf.cond(n > 0.0,
                 lambda: tf.reduce_sum(sq_err) / n,
                 lambda: tf.constant(0.0))


# ---------------------------------------------------------------------------
# TransientHead
# ---------------------------------------------------------------------------

@gin.register
class TransientHead(tf.keras.layers.Layer):
  """Small Conv1D stack that maps z -> per-frame transient labels.

  Input:  z  [B, T, z_dim]
  Output: dict with 6 tensors, each [B, T, 1]:
    onset_logit, onset_offset_norm, onset_velocity,
    offset_logit, offset_offset_norm, silence_logit
  """

  def __init__(self,
               z_dim: int = 48,
               hidden_channels: int = 256,
               kernel_size: int = 3,
               n_layers: int = 3,
               causal: bool = False,
               sample_rate: int = 44100,
               frame_rate: int = 50,
               name: str = 'transient_head',
               **kwargs):
    """
    Args:
      z_dim: Input feature dim (must match encoder z_dims). Default 48.
      hidden_channels: Conv1D width. Default 256.
      kernel_size: Conv1D kernel. Default 3 (1 frame context each side).
      n_layers: Number of Conv1D layers before projection. Default 3.
      causal: If True, use causal convolutions (left-padding only).
              Default False (non-causal, uses ±kernel_size//2 context).
      sample_rate: For normalising onset_offset_norm. Default 44100.
      frame_rate: For normalising onset_offset_norm. Default 50.
    """
    super().__init__(name=name, **kwargs)
    self._causal = causal
    # half_frame_samples used to normalise sub-frame offset to [-1, +1].
    self._half_frame_samples = float(sample_rate) / (2.0 * float(frame_rate))

    padding = 'causal' if causal else 'same'
    layers = []
    for i in range(n_layers):
      in_ch = z_dim if i == 0 else hidden_channels
      layers.append(tf.keras.layers.Conv1D(
          filters=hidden_channels,
          kernel_size=kernel_size,
          padding=padding,
          activation='relu',
          name=f'conv_{i}'))
    self._conv_layers = layers

    # Final projection: 6 output channels, linear activation.
    # Channels in order:
    #   0: onset_logit
    #   1: onset_offset_norm (tanh to constrain to [-1, +1])
    #   2: onset_velocity     (sigmoid to constrain to [0, 1])
    #   3: offset_logit
    #   4: offset_offset_norm (tanh)
    #   5: silence_logit
    self._proj = tf.keras.layers.Conv1D(
        filters=6,
        kernel_size=1,
        padding='same',
        activation=None,
        name='proj')

  def call(self, z: tf.Tensor, training: bool = True) -> dict:
    """Forward pass.

    Args:
      z: [B, T, z_dim] encoder output.
      training: whether in training mode (affects dropout if added later).

    Returns:
      dict with 6 tensors each [B, T, 1].
    """
    x = z
    for layer in self._conv_layers:
      x = layer(x)                                    # [B, T, hidden]

    raw = self._proj(x)                               # [B, T, 6]

    onset_logit        = raw[..., 0:1]
    onset_offset_raw   = raw[..., 1:2]
    onset_velocity_raw = raw[..., 2:3]
    offset_logit       = raw[..., 3:4]
    offset_offset_raw  = raw[..., 4:5]
    silence_logit      = raw[..., 5:6]

    # Constrain regression outputs to valid ranges.
    onset_offset_norm  = tf.tanh(onset_offset_raw)    # [-1, +1]
    onset_velocity     = tf.sigmoid(onset_velocity_raw)  # [0, 1]
    offset_offset_norm = tf.tanh(offset_offset_raw)   # [-1, +1]

    return {
        'onset_logit':        onset_logit,        # [B, T, 1], raw
        'onset_offset_norm':  onset_offset_norm,  # [B, T, 1], tanh
        'onset_velocity':     onset_velocity,     # [B, T, 1], sigmoid
        'offset_logit':       offset_logit,       # [B, T, 1], raw
        'offset_offset_norm': offset_offset_norm, # [B, T, 1], tanh
        'silence_logit':      silence_logit,      # [B, T, 1], raw
    }

  @property
  def half_frame_samples(self) -> float:
    return self._half_frame_samples


# ---------------------------------------------------------------------------
# TransientLoss
# ---------------------------------------------------------------------------

@gin.register
class TransientLoss:
  """Computes and accumulates transient head losses.

  Usable as a standalone loss object (has `get_losses_dict`) — can be
  passed to Autoencoder.losses list or called directly in a model.

  Feature keys expected in the batch dict:
    onset_mask            [B, T] float32
    onset_offset_samples  [B, T] float32  (raw samples, ∈ [-441, +441])
    onset_velocity        [B, T] float32  (normalised 0-1)
    offset_mask           [B, T] float32
    offset_offset_samples [B, T] float32
    silence_mask          [B, T] float32

  These come from the sidecar TFRecord joined by BassWaveWithSidecarProvider.
  """

  def __init__(self,
               lambda_onset: float = 1.0,
               lambda_subframe: float = 0.5,
               lambda_velocity: float = 0.3,
               lambda_offset: float = 0.5,
               lambda_silence: float = 1.5,
               onset_pos_weight: float = 10.0,
               offset_pos_weight: float = 10.0,
               silence_pos_weight: float = 2.0,
               onset_focal_gamma: float = 1.5,
               offset_focal_gamma: float = 1.5,
               name: str = 'transient_loss'):
    self.name = name
    self._lam_on  = float(lambda_onset)
    self._lam_sub = float(lambda_subframe)
    self._lam_vel = float(lambda_velocity)
    self._lam_off = float(lambda_offset)
    self._lam_sil = float(lambda_silence)
    self._on_pw   = float(onset_pos_weight)
    self._off_pw  = float(offset_pos_weight)
    self._sil_pw  = float(silence_pos_weight)
    self._on_gam  = float(onset_focal_gamma)
    self._off_gam = float(offset_focal_gamma)

  def _expand(self, t: tf.Tensor) -> tf.Tensor:
    """Add trailing channel dim if needed: [B, T] -> [B, T, 1]."""
    if len(t.shape) == 2:
      return t[..., tf.newaxis]
    return t

  def get_losses_dict(self, outputs: dict, batch: dict,
                      head: TransientHead) -> dict:
    """Compute all head losses and return a named dict.

    Args:
      outputs: dict from TransientHead.call() — has the 6 prediction tensors.
      batch: the training batch dict — must contain the 6 MIDI target arrays.
      head: TransientHead instance (used for half_frame_samples normalisation).

    Returns:
      Dict mapping loss name -> scalar tf.Tensor. Compatible with
      Model._losses_dict (values summed into 'total_loss').
    """
    # ── Fetch targets from batch (with graceful fallback to zero) ──────────
    def _get(key):
      t = batch.get(key, None)
      if t is None:
        # Sidecar not joined or field missing — return zeros.
        return tf.zeros_like(outputs['silence_logit'])
      return self._expand(tf.cast(t, tf.float32))

    on_mask      = _get('onset_mask')
    on_sub_raw   = _get('onset_offset_samples')
    on_vel_tgt   = _get('onset_velocity')
    off_mask     = _get('offset_mask')
    off_sub_raw  = _get('offset_offset_samples')
    sil_mask     = _get('silence_mask')

    # ── Normalise raw sample offsets to [-1, +1] ───────────────────────────
    hfs = tf.constant(head.half_frame_samples, dtype=tf.float32)
    on_sub_norm  = on_sub_raw  / hfs
    off_sub_norm = off_sub_raw / hfs
    # Clamp to [-1, +1] defensively.
    on_sub_norm  = tf.clip_by_value(on_sub_norm,  -1.0, 1.0)
    off_sub_norm = tf.clip_by_value(off_sub_norm, -1.0, 1.0)

    # ── Compute losses ─────────────────────────────────────────────────────
    l_onset = _weighted_bce(
        outputs['onset_logit'], on_mask,
        self._on_pw, self._on_gam)

    l_sub = _masked_mse(
        outputs['onset_offset_norm'], on_sub_norm, on_mask)

    l_vel = _masked_mse(
        outputs['onset_velocity'], on_vel_tgt, on_mask)

    l_offset = _weighted_bce(
        outputs['offset_logit'], off_mask,
        self._off_pw, self._off_gam)

    l_off_sub = _masked_mse(
        outputs['offset_offset_norm'], off_sub_norm, off_mask)

    l_silence = _weighted_bce(
        outputs['silence_logit'], sil_mask,
        self._sil_pw, 0.0)        # no focal for silence (balanced class)

    losses = {
        'transient/onset':         self._lam_on  * l_onset,
        'transient/subframe_on':   self._lam_sub * l_sub,
        'transient/velocity':      self._lam_vel * l_vel,
        'transient/offset':        self._lam_off * l_offset,
        'transient/subframe_off':  self._lam_sub * l_off_sub,
        'transient/silence':       self._lam_sil * l_silence,
    }
    return losses
