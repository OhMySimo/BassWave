# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Denoising Autoencoder + MIDIHead integration.

Changes vs parent Autoencoder:
  1. Spectral loss target = features['audio_clean'] (denoising semantics).
  2. Optional MIDIHead: predicts per-frame onset/silence/pitch/velocity/ks
     + a 60-dim `midi_cond` tensor consumed as an extra decoder input_key.
  3. Optional pitch fusion (Phase 2 / inference): fuses CREPE f0 with the
     head's pitch prediction to correct octave errors and stabilize.
  4. Optional backbone freeze: Phase-1 warm-up freezes encoder+decoder so
     only the head trains; midi_cond is teacher-forced from MIDI GT.
  5. Inference silence gate (audio-rate): final audio_synth multiplied by
     (1 - silence_prob) upsampled to audio rate.

Phases
------
Phase 1 — Warm-up (freeze_backbone=True, use_teacher_forcing_midi=True):
  • encoder + decoder frozen at checkpoint v3 weights
  • MIDIHead trained alone on midi-* losses
  • decoder receives teacher-forced midi_cond (= MIDI GT)
  • spectral loss still computed but ∂L_spec/∂params = 0 on frozen layers
    (logged only — useful to track if midi_cond is actually used by
    decoder when later unfrozen)

Phase 2 — Joint (freeze_backbone=False, use_teacher_forcing_midi=False,
                  use_pitch_fusion=True):
  • all trainable
  • decoder consumes midi_cond predicted by head
  • f0_hz / f0_scaled overridden by pitch fusion before decoder
  • all losses active

Note on gin registration: this class is registered exactly once by
models/__init__.py via _configurable(DenoisingAutoencoder). Don't add
@gin.register here.
"""
from __future__ import annotations

import ddsp
import tensorflow.compat.v2 as tf

from ddsp.training import preprocessing
from ddsp.training.midi_head import MIDIHead, fuse_pitch_with_crepe
from ddsp.training.models.autoencoder import Autoencoder


class DenoisingAutoencoder(Autoencoder):
  """Autoencoder with denoising loss target + optional MIDIHead conditioning."""

  def __init__(self,
               preprocessor=None,
               encoder=None,
               decoder=None,
               processor_group=None,
               losses=None,
               # MIDI head wiring.
               mel_extractor=None,
               midi_head=None,
               midi_head_loss_obj=None,
               # Z-aux self-distillation wiring.
               z_aux_projector=None,
               z_aux_loss_obj=None,
               # Training phase controls.
               freeze_backbone: bool = False,
               use_teacher_forcing_midi: bool = False,
               # Inference / phase 2 helpers.
               use_pitch_fusion: bool = False,
               use_silence_gate: bool = True,
               silence_gate_floor: float = 0.0,
               **kwargs):
    """
    Args:
      mel_extractor: LogMelExtractor (or None to disable MIDI head pathway).
      midi_head: MIDIHead instance (or None).
      midi_head_loss_obj: MIDIHeadLoss instance (required when head present
        AND training=True).
      z_aux_projector: ZAuxProjector instance (or None). When provided,
        projects midi_cond \u2192 z_aux at every forward and contributes
        z_aux_loss to total. Used for self-distillation training. Does NOT
        feed into the decoder (decoder remains input_keys=3 = P1 compatible).
      z_aux_loss_obj: ZAuxDistillationLoss instance. Required when
        z_aux_projector is provided AND training=True.
      freeze_backbone: if True, sets encoder.trainable = decoder.trainable
        = False. Used in Phase 1.
      use_teacher_forcing_midi: if True, decoder receives midi_cond built
        from MIDI GT instead of head predictions. Phase 1.
      use_pitch_fusion: if True, override features['f0_hz']/['f0_scaled']
        with fuse_pitch_with_crepe before decoder. Phase 2 / inference.
      use_silence_gate: if True, multiply audio_synth by (1 - sigmoid(
        silence_logit)) at audio rate during inference (training=False).
      silence_gate_floor: minimum gate value (0=full mute, e.g. 0.05 = -26 dB).
    """
    super().__init__(
        preprocessor=preprocessor,
        encoder=encoder,
        decoder=decoder,
        processor_group=processor_group,
        losses=losses,
        **kwargs)
    self.mel_extractor = mel_extractor
    self.midi_head = midi_head
    self.midi_head_loss_obj = midi_head_loss_obj
    self.z_aux_projector = z_aux_projector
    self.z_aux_loss_obj = z_aux_loss_obj
    self._freeze_backbone = bool(freeze_backbone)
    self._use_teacher_forcing = bool(use_teacher_forcing_midi)
    self._use_pitch_fusion = bool(use_pitch_fusion)
    self._use_silence_gate = bool(use_silence_gate)
    self._silence_gate_floor = float(silence_gate_floor)
    self._backbone_frozen_applied = False  # set lazily on first call

  # ───────────────────────────────────────────────────────────────────────
  # Frozen-backbone wiring.
  # ───────────────────────────────────────────────────────────────────────

  def _maybe_freeze_backbone(self):
    """Toggle trainable=False on encoder+decoder once.

    Called lazily on first forward pass. Keras requires the variables to
    exist (which happens after the first call), so we can't freeze in
    __init__.
    """
    if self._backbone_frozen_applied or not self._freeze_backbone:
      return
    if self.encoder is not None:
      self.encoder.trainable = False
    if self.decoder is not None:
      self.decoder.trainable = False
    self._backbone_frozen_applied = True

  # ───────────────────────────────────────────────────────────────────────
  # Head plumbing.
  # ───────────────────────────────────────────────────────────────────────

  def _run_midi_head(self, features: dict, training: bool) -> dict:
    """Run MIDIHead on audio_16k → log_mel → predictions.

    Uses audio_16k (always present in our batches) for consistency with
    training/inference. Skips silently if components missing.
    """
    if self.midi_head is None or self.mel_extractor is None:
      return {}
    if 'audio_16k' not in features:
      return {}
    audio_16k = features['audio_16k']
    log_mel = self.mel_extractor(audio_16k)                 # [B, T, n_mels]
    
    # Align log_mel time steps with target features to fix STFT padding mismatches
    if 'onset_mask' in features:
      target_len = features['onset_mask'].shape[1] or tf.shape(features['onset_mask'])[1]
      log_mel = log_mel[:, :target_len, :]
    elif 'f0_hz' in features:
      target_len = features['f0_hz'].shape[1] or tf.shape(features['f0_hz'])[1]
      log_mel = log_mel[:, :target_len, :]

    return self.midi_head(log_mel, training=training)

  def _resolve_midi_cond(self, head_outputs: dict, features: dict) -> tf.Tensor:
    """Pick midi_cond source: teacher-forced from GT, or head predictions.

    During Phase 1 warm-up, decoder must receive GT-quality midi_cond.
    During Phase 2 + inference, use head predictions.
    """
    if self._use_teacher_forcing and 'onset_mask' in features:
      return MIDIHead.build_teacher_midi_cond(features)
    if head_outputs and 'midi_cond' in head_outputs:
      return head_outputs['midi_cond']
    return None

  def _maybe_fuse_pitch(self, features: dict, head_outputs: dict):
    """Override features['f0_hz'] and features['f0_scaled'] via head fusion.

    No-op if pitch fusion disabled, or head missing required outputs, or
    f0 features unavailable.
    """
    if not self._use_pitch_fusion or not head_outputs:
      return features
    if ('pitch_logits' not in head_outputs
        or 'silence_logit' not in head_outputs):
      return features
    if 'f0_hz' not in features or 'f0_confidence' not in features:
      return features

    f0_hz = features['f0_hz']
    conf = features['f0_confidence']
    f0_fused = fuse_pitch_with_crepe(
        f0_crepe=f0_hz,
        f0_crepe_conf=conf,
        pitch_logits_head=head_outputs['pitch_logits'],
        silence_logit_head=head_outputs['silence_logit'],
        onset_logit_head=head_outputs.get('onset_logit', None))
    features['f0_hz'] = f0_fused
    # f0_scaled must be recomputed by scale_f0_hz to stay coherent.
    features['f0_scaled'] = preprocessing.scale_f0_hz(f0_fused)
    return features

  def apply_audio_rate_silence_gate(self,
                                    audio_synth: tf.Tensor,
                                    silence_logit: tf.Tensor) -> tf.Tensor:
    """Multiply audio_synth by (1 - silence_prob) upsampled to audio rate."""
    silence_prob = tf.sigmoid(silence_logit)                # [B, Tf, 1]
    gate_frames = 1.0 - silence_prob
    if self._silence_gate_floor > 0.0:
      gate_frames = tf.maximum(gate_frames, self._silence_gate_floor)
    n_audio = tf.shape(audio_synth)[-1]
    gate_audio = ddsp.core.resample(gate_frames, n_audio)   # [B, Ta, 1]
    gate_audio = tf.squeeze(gate_audio, axis=-1)
    return audio_synth * gate_audio

  # ───────────────────────────────────────────────────────────────────────
  # Forward.
  # ───────────────────────────────────────────────────────────────────────

  def call(self, features, training=True):
    """Encode → MIDI head → (maybe pitch fuse) → decode → losses / gate."""
    self._maybe_freeze_backbone()

    # 1. Encode.
    features = self.encode(features, training=training)

    # 2. MIDI head (runs on audio_16k always — encoder-z-independent).
    head_outputs = self._run_midi_head(features, training=training)

    # 3. Pitch fusion (Phase 2 / inference).
    features = self._maybe_fuse_pitch(features, head_outputs)

    # 4. midi_cond resolution + injection into features for decoder.
    midi_cond = self._resolve_midi_cond(head_outputs, features)
    if midi_cond is not None:
      features['midi_cond'] = midi_cond

    # 5. Z-aux self-distillation: project midi_cond -> z_aux.
    # Crucially this projector is SEPARATE from the decoder path. midi_cond
    # is fed ONLY to the projector. The decoder remains input_keys=3 (P1
    # compatible). The aux_loss regularizes z by aligning it with what the
    # projector can predict from MIDI knowledge.
    z_aux = None
    if (self.z_aux_projector is not None and midi_cond is not None
        and 'z' in features):
      z_aux = self.z_aux_projector(midi_cond, training=training)

    # 6. Decoder + processor group.
    features.update(self.decoder(features, training=training))
    pg_out = self.processor_group(features, return_outputs_dict=True)
    outputs = pg_out['controls']
    outputs['audio_synth'] = pg_out['signal']

    # Logging stash.
    outputs['audio_corrupted'] = features['audio']
    outputs['audio_clean'] = features.get('audio_clean', features['audio'])
    outputs.update(head_outputs)
    if z_aux is not None:
      outputs['z_aux'] = z_aux

    if training:
      # Spectral loss vs clean target.
      target = features.get('audio_clean', features['audio'])
      self._update_losses_dict(
          self.loss_objs, target, outputs['audio_synth'])
      # MIDI head losses.
      if (self.midi_head is not None
          and self.midi_head_loss_obj is not None
          and head_outputs):
        head_losses = self.midi_head_loss_obj.get_losses_dict(
            head_outputs=head_outputs, batch=features)
        self._losses_dict.update(head_losses)
      # Z-aux distillation loss.
      if (self.z_aux_projector is not None
          and self.z_aux_loss_obj is not None
          and z_aux is not None):
        z_aux_losses = self.z_aux_loss_obj.get_losses_dict(
            z=features['z'], z_aux=z_aux)
        self._losses_dict.update(z_aux_losses)
    else:
      # Inference: audio-rate silence gate.
      if (self._use_silence_gate and 'silence_logit' in head_outputs):
        outputs['audio_synth'] = self.apply_audio_rate_silence_gate(
            outputs['audio_synth'],
            head_outputs['silence_logit'])

    return outputs

  # ───────────────────────────────────────────────────────────────────────
  # Inference helper for preset z override.
  # ───────────────────────────────────────────────────────────────────────

  def call_with_preset(self,
                       features: dict,
                       z_preset: tf.Tensor = None,
                       use_silence_gate: bool = None,
                       use_pitch_fusion: bool = None) -> dict:
    """Inference: head sees encoder-z; decoder uses z_preset for timbre.

    Order:
      encode → midi_head(audio_16k → mel) → pitch_fuse → z=z_preset →
      decoder → processor_group → silence_gate
    """
    if use_silence_gate is None:
      use_silence_gate = self._use_silence_gate
    if use_pitch_fusion is None:
      use_pitch_fusion = self._use_pitch_fusion

    features = self.encode(features, training=False)
    head_outputs = self._run_midi_head(features, training=False)

    if use_pitch_fusion:
      features = self._maybe_fuse_pitch(features, head_outputs)

    midi_cond = self._resolve_midi_cond(head_outputs, features)
    if midi_cond is not None:
      features['midi_cond'] = midi_cond

    # z override for timbre transfer.
    if z_preset is not None:
      features['z'] = z_preset

    features.update(self.decoder(features, training=False))
    pg_out = self.processor_group(features, return_outputs_dict=True)
    outputs = pg_out['controls']
    outputs['audio_synth'] = pg_out['signal']
    outputs.update(head_outputs)

    if use_silence_gate and 'silence_logit' in head_outputs:
      outputs['audio_synth'] = self.apply_audio_rate_silence_gate(
          outputs['audio_synth'],
          head_outputs['silence_logit'])
      outputs['silence_gate_frames'] = 1.0 - tf.sigmoid(
          head_outputs['silence_logit'])

    return outputs
