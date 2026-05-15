# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Z-aux self-distillation: head MIDI knowledge → encoder z.

Concept
-------
La head MIDI sa molte cose che CREPE non sa: onset timing, silence boundary,
pitch class robusta a degradation. Vogliamo che l'encoder produca un `z`
strutturalmente coerente con quella conoscenza, SENZA che la head condizioni
direttamente il decoder (che si \u00e8 rotto in Phase 2 con quell'approccio).

Implementazione
---------------
Aggiungiamo un piccolo "projector" (~30k params, MLP 60 → 64 → 48) che impara
a mappare:
  midi_cond [B, T, 60]  →  z_aux [B, T, 48]

Loss aux = MSE(z, z_aux). Gradient flow naturale:
  • verso projector: impara a predire z da midi_cond
  • verso encoder: impara a produrre z predicibile dalla MIDI knowledge
  • la head non riceve gradient da questa loss (midi_cond gi\u00e0 trained in P1,
    e qui passa solo come INPUT al projector, non come supervisione)

Bilanciamento col main spectral loss
------------------------------------
La spectral_loss del decoder \u00e8 il segnale primario di apprendimento. La
aux_loss agisce come regolarizzatore: spinge z verso un manifold pi\u00f9
musicalmente strutturato. Lambda critico: troppo alto e l'encoder fa
collapse (produce z banali predicibili da midi_cond, perde info timbric);
troppo basso e l'effetto regolarizzante svanisce.

Default lambda_aux=0.05 (basso, conservativo).
"""
from __future__ import annotations

import gin
import tensorflow.compat.v2 as tf


# ─────────────────────────────────────────────────────────────────────────────
# Projector
# ─────────────────────────────────────────────────────────────────────────────

@gin.register
class ZAuxProjector(tf.keras.layers.Layer):
  """Maps midi_cond [B, T, midi_cond_dim] -> z_aux [B, T, z_dim].

  Small MLP applied per-frame (Conv1D kernel=1 = TimeDistributed Dense),
  with a residual smoothing pass to encourage temporal coherence.
  """

  def __init__(self,
               midi_cond_dim: int = 60,
               z_dim: int = 48,
               hidden_dim: int = 64,
               use_temporal_smoothing: bool = True,
               name: str = 'zaux_projector',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.midi_cond_dim = midi_cond_dim
    self.z_dim = z_dim
    self.hidden_dim = hidden_dim
    self.use_temporal_smoothing = use_temporal_smoothing

    # Per-frame MLP.
    self.dense1 = tf.keras.layers.Conv1D(
        hidden_dim, kernel_size=1, padding='same', activation='gelu',
        name='dense1')
    self.dense2 = tf.keras.layers.Conv1D(
        z_dim, kernel_size=1, padding='same', activation=None,
        name='dense2')

    # Light temporal smoothing (k=5 conv) so that z_aux predictions
    # leverage local context — onset frames affect neighbours.
    if use_temporal_smoothing:
      self.smooth = tf.keras.layers.Conv1D(
          z_dim, kernel_size=5, padding='same', activation=None,
          name='smooth')

  def call(self, midi_cond: tf.Tensor, training: bool = True) -> tf.Tensor:
    """midi_cond [B, T, midi_cond_dim] -> z_aux [B, T, z_dim]."""
    x = self.dense1(midi_cond)
    z_aux = self.dense2(x)
    if self.use_temporal_smoothing:
      # Residual: z_aux + smooth(z_aux).
      z_aux = z_aux + self.smooth(z_aux)
    return z_aux


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

@gin.register
class ZAuxDistillationLoss:
  """Scale-invariant MSE between encoder z and projected z_aux from midi_cond.

  Standard MSE(z, z_aux) is dominated by the *magnitude* of z, which can be
  small at init (z in ~[-1, +1], z_aux near 0 because Conv1D bias=0). With
  z range ~1 and z_aux ~0, raw MSE is ~0.1, lambda*MSE ~0.005 — gradient
  too small to meaningfully push the encoder.

  SCALE-INVARIANT LOSS:
    nmse = MSE(z, z_aux) / (var(z) + epsilon)
  
  At step 0: z_aux ~ 0 → MSE ~ var(z) → nmse ~ 1.0 (clean signal).
  At convergence: z_aux ~ z → nmse ~ 0.
  
  This decouples the loss magnitude from the absolute scale of z and gives
  a clean [0..1] regularization signal regardless of how the encoder
  normalizes its output. lambda_aux can then be set to a meaningful value
  (e.g. 0.3 = aux contributes ~30% as much as a perfectly-converged loss
  to total when nmse=1, scaling down as nmse decreases).

  Both directions are open (no stop_gradient):
    • encoder learns to produce z aligned with what projector can predict
    • projector learns to predict z from midi_cond

  Convergence: both meet at a point where z lives in a manifold predictable
  from MIDI semantics. Encoder retains freedom to encode timbric info not
  captured by midi_cond because spectral_loss (primary) forces utility.
  """

  def __init__(self,
               lambda_aux: float = 0.3,
               freeze_after_step: int = -1,
               epsilon: float = 1e-3,
               name: str = 'z_aux_distill_loss'):
    """
    Args:
      lambda_aux: weight of the aux loss. With scale-invariant formulation
        a reasonable range is 0.1..0.5. Default 0.3.
      freeze_after_step: if > 0, weight becomes 0 after this step. Default -1.
      epsilon: numerical stability for var(z) denominator. Default 1e-3.
    """
    self.name = name
    self._lambda = float(lambda_aux)
    self._freeze_step = int(freeze_after_step)
    self._eps = float(epsilon)

  def get_losses_dict(self, z: tf.Tensor, z_aux: tf.Tensor,
                      step: tf.Tensor = None) -> dict:
    """
    Args:
      z:     [B, T, z_dim] from encoder.
      z_aux: [B, T, z_dim] from projector.
      step:  optional scalar tf.Tensor of current global step.

    Returns dict with:
      'z_aux/nmse': weight * normalized_mse  (scale-invariant loss for total)
        At step 0 (z_aux ~ 0): nmse ~ 1.0 → contributes ~ lambda_aux
        At convergence (z_aux ~ z): nmse ~ 0 → contributes ~ 0
    """
    # Raw MSE.
    sq_diff = tf.square(z - z_aux)
    mse = tf.reduce_mean(sq_diff)

    # Variance of z (treated as constant for the encoder's gradient on the
    # *denominator*: we want gradient through z primarily via the numerator,
    # otherwise the encoder could trivially minimize loss by increasing
    # var(z) without aligning to z_aux).
    z_mean = tf.reduce_mean(z, axis=[0, 1], keepdims=True)
    var_z = tf.reduce_mean(tf.square(z - z_mean))
    var_z_sg = tf.stop_gradient(var_z)

    # Scale-invariant MSE.
    nmse = mse / (var_z_sg + self._eps)

    # Scheduled weight.
    weight = tf.constant(self._lambda, dtype=tf.float32)
    if self._freeze_step > 0 and step is not None:
      active = tf.cast(step < self._freeze_step, tf.float32)
      weight = weight * active

    return {
        # Only nmse contributes to total. raw_mse and var_z are recoverable
        # from nmse and var_z if needed but we don't return them here because
        # DDSP's Model.sum_losses() sums ALL entries of _losses_dict into
        # total_loss; adding raw_mse would double-count and adding var_z
        # would push the encoder to MINIMIZE its own variance (collapse).
        'z_aux/nmse': weight * nmse,
    }
