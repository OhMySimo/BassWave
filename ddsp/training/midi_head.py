# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""MIDI Head — U-Net 1D che dall'audio mel-spec predice onset / silence /
pitch / velocity / key-switches per frame, e produce midi_cond [B, T, 60]
usato come input aggiuntivo del decoder.

Architettura (U-Net 1D 3-level, ~4.2M params):

  audio_16k [B, 64000]
    └→ LogMelExtractor → log_mel [B, T=200, n_mels=128]
            ↓
       enc1a/b (256ch, dil 1+2)              → skip1 [B, 200, 256]
            ↓ down ×2
       enc2a/b (384ch, dil 1+2)              → skip2 [B, 100, 384]
            ↓ down ×2
       bottleneck (512ch, dil 4+8) + BiGRU(256↔256=512)
            ↓ up ×2 + concat(skip2)
       dec2 (384ch)
            ↓ up ×2 + concat(skip1)
       dec1 (256ch)
            ↓
       5 head branches:
         onset_head:   [B, T, 2]  (logit + subframe_norm)
         silence_head: [B, T, 1]
         pitch_head:   [B, T, 45] (0=silence, 1..44 = MIDI 21..64)
         velocity_head:[B, T, 1]  (sigmoid)
         ks_head:      [B, T, 12] (multi-label sigmoid)
            ↓
       midi_cond [B, T, 60] = concat of probabilities
                            (semanticamente significativo,
                             permette teacher-forcing diretto da MIDI GT).

Pitch fusion (a inferenza + Phase 2 training):
  Vedi `fuse_pitch_with_crepe()` per logica 3-stage:
    1. Octave-error correction (CREPE su 2× o 0.5× del reale)
    2. Confidence-weighted blend (alpha = sigmoid(8*(conf - 0.5)))
    3. Silence handling (skip fusion quando head predice silenzio)
"""
from __future__ import annotations

import gin
import tensorflow.compat.v2 as tf


# ─────────────────────────────────────────────────────────────────────────────
# Mel-spectrogram extraction
# ─────────────────────────────────────────────────────────────────────────────

@gin.register
class LogMelExtractor(tf.keras.layers.Layer):
  """Estrae log-mel-spectrogram da audio mono.

  Input  : audio [B, T_audio]  (es. 64000 = 16 kHz × 4 s)
  Output : log_mel [B, T, n_mels]  (es. 200 frame × 128 mels)

  Frame-rate target: 50 Hz @ 16 kHz → hop = 320, win = 1024.
  Centred=True (REFLECT pad) per matchare il framing di prepare_basswave.
  """

  def __init__(self,
               sample_rate: int = 16000,
               n_mels: int = 128,
               win_length: int = 1024,
               hop_length: int = 320,
               fmin: float = 40.0,
               fmax: float = 8000.0,
               log_offset: float = 1e-6,
               name: str = 'log_mel_extractor',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.sample_rate = sample_rate
    self.n_mels = n_mels
    self.win_length = win_length
    self.hop_length = hop_length
    self.fmin = fmin
    self.fmax = fmax
    self.log_offset = log_offset

    # FFT length: next power of 2 ≥ win_length.
    nfft = 1
    while nfft < win_length:
      nfft *= 2
    self._nfft = nfft
    n_freq = nfft // 2 + 1
    self._mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_freq,
        sample_rate=sample_rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )

  def call(self, audio: tf.Tensor) -> tf.Tensor:
    # Centred framing via REFLECT pad.
    pad = self.win_length // 2
    audio_p = tf.pad(audio, [[0, 0], [pad, pad]], mode='REFLECT')
    spec = tf.signal.stft(
        audio_p,
        frame_length=self.win_length,
        frame_step=self.hop_length,
        fft_length=self._nfft,
        pad_end=False)                                  # [B, T, n_freq]
    mag = tf.abs(spec)
    mel = tf.matmul(mag * mag, self._mel_w)             # [B, T, n_mels]
    return tf.math.log(mel + self.log_offset)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBlock(tf.keras.layers.Layer):
  """Conv1D + GroupNorm + GELU."""

  def __init__(self, channels, kernel_size=5, dilation=1, gn_groups=8,
               name='conv_block', **kwargs):
    super().__init__(name=name, **kwargs)
    self.conv = tf.keras.layers.Conv1D(
        filters=channels, kernel_size=kernel_size, dilation_rate=dilation,
        padding='same', activation=None, name='conv')
    # GroupNorm: gn_groups must divide channels. Auto-adjust.
    g = gn_groups
    while channels % g != 0 and g > 1:
      g -= 1
    self.gn = tf.keras.layers.GroupNormalization(
        groups=g, axis=-1, name='gn')
    self.act = tf.keras.layers.Activation('gelu', name='gelu')

  def call(self, x, training=True):
    return self.act(self.gn(self.conv(x), training=training))


class _DownBlock(tf.keras.layers.Layer):
  """Stride-2 downsample."""

  def __init__(self, channels, name='down', **kwargs):
    super().__init__(name=name, **kwargs)
    self.down = tf.keras.layers.Conv1D(
        filters=channels, kernel_size=4, strides=2, padding='same',
        activation=None, name='down_conv')

  def call(self, x):
    return self.down(x)


class _UpBlock(tf.keras.layers.Layer):
  """Linear upsample ×2 + Conv1D."""

  def __init__(self, channels, name='up', **kwargs):
    super().__init__(name=name, **kwargs)
    self.up = tf.keras.layers.UpSampling1D(size=2, name='up_x2')
    self.conv = tf.keras.layers.Conv1D(
        filters=channels, kernel_size=3, padding='same',
        activation=None, name='up_conv')

  def call(self, x):
    return self.conv(self.up(x))


# ─────────────────────────────────────────────────────────────────────────────
# MIDIHead — U-Net 1D
# ─────────────────────────────────────────────────────────────────────────────

@gin.register
class MIDIHead(tf.keras.layers.Layer):
  """U-Net 1D: log-mel → 5 prediction heads + midi_cond [B, T, 60]."""

  N_PITCH_CLASSES = 45     # 0=silence, 1..44 = MIDI 21..64
  N_KS = 12
  MIDI_PITCH_MIN = 21
  MIDI_PITCH_MAX = 64
  MIDI_COND_DIM = 60       # = 1 (onset) + 1 (silence) + 45 (pitch) + 1 (vel) + 12 (ks)

  # Maschera dei bit KS che portano segnale musicale reale.
  # bit 3 = KS_NOT_USED (midi_id 12, trigger null) → azzerato ovunque.
  # Shape [12] float32; moltiplicata su ks_bits prima di concat in midi_cond
  # e prima del calcolo della loss, così il modello non sprechi capacità su
  # un neurone sempre-0.
  KS_ACTIVE_MASK = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # bit 3 = 0

  def __init__(self,
               n_mels: int = 128,
               ch_d1: int = 256,
               ch_d2: int = 384,
               ch_bottleneck: int = 512,
               bigru_units: int = 256,
               name: str = 'midi_head',
               **kwargs):
    super().__init__(name=name, **kwargs)

    # Down path.
    self.enc1a = _ConvBlock(ch_d1, kernel_size=5, dilation=1, name='enc1a')
    self.enc1b = _ConvBlock(ch_d1, kernel_size=5, dilation=2, name='enc1b')
    self.down1 = _DownBlock(ch_d1, name='down1')
    self.enc2a = _ConvBlock(ch_d2, kernel_size=5, dilation=1, name='enc2a')
    self.enc2b = _ConvBlock(ch_d2, kernel_size=5, dilation=2, name='enc2b')
    self.down2 = _DownBlock(ch_d2, name='down2')

    # Bottleneck.
    self.bottleneck1 = _ConvBlock(
        ch_bottleneck, kernel_size=5, dilation=4, name='bottle1')
    self.bottleneck2 = _ConvBlock(
        ch_bottleneck, kernel_size=5, dilation=8, name='bottle2')
    self.bigru = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            units=bigru_units, return_sequences=True, name='gru'),
        merge_mode='concat', name='bigru')

    # Up path.
    self.up2 = _UpBlock(ch_d2, name='up2')
    self.dec2 = _ConvBlock(ch_d2, kernel_size=5, name='dec2')
    self.up1 = _UpBlock(ch_d1, name='up1')
    self.dec1 = _ConvBlock(ch_d1, kernel_size=5, name='dec1')

    # Heads.
    self.onset_pre = tf.keras.layers.Conv1D(
        64, 3, padding='same', activation='gelu', name='onset_pre')
    self.onset_out = tf.keras.layers.Conv1D(
        2, 1, padding='same', name='onset_out')
    self.silence_pre = tf.keras.layers.Conv1D(
        64, 3, padding='same', activation='gelu', name='sil_pre')
    self.silence_out = tf.keras.layers.Conv1D(
        1, 1, padding='same', name='sil_out')
    self.pitch_pre = tf.keras.layers.Conv1D(
        128, 3, padding='same', activation='gelu', name='pitch_pre')
    self.pitch_out = tf.keras.layers.Conv1D(
        self.N_PITCH_CLASSES, 1, padding='same', name='pitch_out')
    self.velocity_pre = tf.keras.layers.Conv1D(
        64, 3, padding='same', activation='gelu', name='vel_pre')
    self.velocity_out = tf.keras.layers.Conv1D(
        1, 1, padding='same', name='vel_out')
    self.ks_pre = tf.keras.layers.Conv1D(
        64, 3, padding='same', activation='gelu', name='ks_pre')
    self.ks_out = tf.keras.layers.Conv1D(
        self.N_KS, 1, padding='same', name='ks_out')

  def call(self, log_mel: tf.Tensor, training: bool = True) -> dict:
    """
    Args:
      log_mel: [B, T=200, n_mels=128]

    Returns:
      dict (see module docstring for shapes).
    """
    x = self.enc1a(log_mel, training=training)
    skip1 = self.enc1b(x, training=training)
    x = self.down1(skip1)
    x = self.enc2a(x, training=training)
    skip2 = self.enc2b(x, training=training)
    x = self.down2(skip2)

    x = self.bottleneck1(x, training=training)
    x = self.bottleneck2(x, training=training)
    x = self.bigru(x, training=training)

    x = self.up2(x)
    # Crop temporal dim to skip2 in case input T is odd (e.g. 201 → down=101
    # → up=102 but skip has 101). Standard U-Net fix for non-power-of-2 T.
    x = x[:, :tf.shape(skip2)[1], :]
    x = tf.concat([x, skip2], axis=-1)
    x = self.dec2(x, training=training)
    x = self.up1(x)
    # Same fix for level-1 skip.
    x = x[:, :tf.shape(skip1)[1], :]
    x = tf.concat([x, skip1], axis=-1)
    x = self.dec1(x, training=training)

    onset_raw = self.onset_out(self.onset_pre(x))
    onset_logit = onset_raw[..., 0:1]
    onset_subframe = tf.tanh(onset_raw[..., 1:2])

    silence_logit = self.silence_out(self.silence_pre(x))
    pitch_logits = self.pitch_out(self.pitch_pre(x))
    velocity = tf.sigmoid(self.velocity_out(self.velocity_pre(x)))
    ks_logits = self.ks_out(self.ks_pre(x))

    # midi_cond (60 dims): probabilità interpretabili.
    onset_prob = tf.sigmoid(onset_logit)
    silence_prob = tf.sigmoid(silence_logit)
    pitch_probs = tf.nn.softmax(pitch_logits, axis=-1)
    ks_probs = tf.sigmoid(ks_logits)
    midi_cond = tf.concat(
        [onset_prob, silence_prob, pitch_probs, velocity, ks_probs],
        axis=-1)                                            # [B, T, 60]

    return {
        'onset_logit': onset_logit,
        'onset_subframe': onset_subframe,
        'silence_logit': silence_logit,
        'pitch_logits': pitch_logits,
        'velocity': velocity,
        'ks_logits': ks_logits,
        'midi_cond': midi_cond,
    }

  # ─────────────────────────────────────────────────────────────────────────
  # Teacher-forcing midi_cond constructor.
  # ─────────────────────────────────────────────────────────────────────────

  @staticmethod
  def build_teacher_midi_cond(batch: dict) -> tf.Tensor:
    """midi_cond [B, T, 60] direttamente dai MIDI targets.

    Usato in Phase 1 (warm-up) per fornire al decoder midi_cond perfetto
    invece di predictions garbage da head random-initialized. Dim coerente
    con MIDIHead.call() output -> decoder shape invariata in Phase 1 vs 2.
    """
    onset = batch['onset_mask'][..., tf.newaxis]
    silence = batch['silence_mask'][..., tf.newaxis]
    velocity = batch['onset_velocity'][..., tf.newaxis]

    # Pitch class: 0=silence, k≥1 -> MIDI = k + 20.
    note_midi = tf.cast(batch['active_note_midi'], tf.int32)
    pitch_class = tf.where(
        note_midi == 0,
        tf.zeros_like(note_midi),
        note_midi - (MIDIHead.MIDI_PITCH_MIN - 1))
    pitch_class = tf.clip_by_value(
        pitch_class, 0, MIDIHead.N_PITCH_CLASSES - 1)
    pitch_onehot = tf.one_hot(
        pitch_class, MIDIHead.N_PITCH_CLASSES, dtype=tf.float32)

    # KS bits: bit-decomposition of active_ks_bits.
    # KS_ACTIVE_MASK azzera bit 3 (KS_NOT_USED) coerentemente con la loss.
    ks_int = tf.cast(batch['active_ks_bits'], tf.int32)
    bits = [tf.cast(
        tf.bitwise.right_shift(ks_int, i) & 1, tf.float32)
        for i in range(MIDIHead.N_KS)]
    ks_mask = tf.constant(MIDIHead.KS_ACTIVE_MASK, dtype=tf.float32)
    ks_bits = tf.stack(bits, axis=-1) * ks_mask

    return tf.concat(
        [onset, silence, pitch_onehot, velocity, ks_bits], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Pitch fusion utility (inference-time)
# ─────────────────────────────────────────────────────────────────────────────

def fuse_pitch_with_crepe(f0_crepe: tf.Tensor,
                          f0_crepe_conf: tf.Tensor,
                          pitch_logits_head: tf.Tensor,
                          silence_logit_head: tf.Tensor,
                          onset_logit_head: tf.Tensor = None,
                          onset_threshold: float = 0.5,
                          silence_threshold: float = 0.5,
                          ) -> tf.Tensor:
  """Onset-segmented pitch-class-only fusion.

  Concept
  -------
  La head detta CONFINI DI NOTA (onset → onset successivo) e PITCH CLASS
  (C..B, 12 classi). CREPE detta OTTAVA + continuum intra-nota (slides).
  
  Procedura per ogni frame i:
    1. Partiziona la sequenza in segmenti via onset_logit (onset peak → onset
       successivo). Ogni segmento = 1 nota.
    2. Per ogni segmento determina la pitch class dominante della head
       (modal class su quel segmento, escludendo silence frame).
    3. Per ogni frame i del segmento:
         midi_crepe[i] = continuous MIDI da f0_crepe
         octave[i]     = round(midi_crepe[i] / 12)
         midi_fused[i] = octave[i] * 12 + pc_segment
         Aggiusta se |midi_fused - midi_crepe| > 6 (snap a ottava piu' vicina)
    4. f0_fused[i] = 440 * 2^((midi_fused[i] - 69) / 12)
  
  Slides: dentro un segmento midi_fused puo' cambiare semitone come fa
  CREPE, ma SOLO sui boundaries della pitch class scelta. Es. se pc=FA e
  CREPE attraversa FA-FA#-SOL, midi_fused resta su FA finche' CREPE non
  e' piu' vicino a FA#, poi salta. In pratica per slide veri (mezzo step
  per nota) la fusion preserva ogni semitone.
  
  Silence
  -------
  Se onset_logit_head NON e' fornito, fallback a un comportamento per-frame
  (no segmentazione): in frame silence -> f0_crepe pass-through, altrimenti
  apply pc snap con pitch class instantanea della head.

  Args:
    f0_crepe:           [B, T] o [B, T, 1] in Hz.
    f0_crepe_conf:      [B, T] o [B, T, 1] in [0, 1] — non usato in questa
                        versione (la head e' considerata sempre piu'
                        affidabile di CREPE per la classe), ma tenuto in
                        signature per compat retroattiva.
    pitch_logits_head:  [B, T, 45] raw head pitch logits.
    silence_logit_head: [B, T, 1] raw silence logit.
    onset_logit_head:   [B, T, 1] raw onset logit. Se None, segmentazione
                        per-frame.
    onset_threshold:    soglia sigmoid per onset detection.
    silence_threshold:  soglia sigmoid per silence detection.

  Returns:
    f0_fused: stessa shape di f0_crepe (Hz).
  """
  del f0_crepe_conf  # unused, kept for signature compat

  # Normalize shape to [B, T, 1] internally, then squeeze back at the end
  # if input was 2D.
  if len(f0_crepe.shape) == 2:
    f0_c = f0_crepe[..., tf.newaxis]
    squeeze_back = True
  else:
    f0_c = f0_crepe
    squeeze_back = False

  silence_prob = tf.sigmoid(silence_logit_head)            # [B, T, 1]
  is_silence = silence_prob > silence_threshold            # [B, T, 1] bool

  # ── Pitch class per frame from head ─────────────────────────────────
  # MIDI class space:
  #   class 0 = silence
  #   class k>0 = MIDI pitch (k + MIDI_PITCH_MIN - 1) for k in 1..44
  # We need pitch_class (0..11) = MIDI % 12. Soft argmax over the active
  # classes only.
  pitch_probs = tf.nn.softmax(pitch_logits_head, axis=-1)        # [B, T, 45]
  # Build a [N_PITCH_CLASSES] lookup: for class k=0 → pc=-1 (sentinel);
  # for class k=1..44 → pc = (k + MIN - 1) % 12.
  classes = tf.range(MIDIHead.N_PITCH_CLASSES, dtype=tf.int32)   # [45]
  pc_table = tf.where(
      classes == 0,
      -tf.ones_like(classes),                                    # silence sentinel
      tf.math.floormod(classes + (MIDIHead.MIDI_PITCH_MIN - 1), 12))
  # Per frame, weighted mode over pc_table (using head probs).
  # For each frame, accumulate prob mass into each of the 12 pc bins, then
  # argmax → integer pitch class. Vectorised via a [45, 12] one-hot of
  # pc_table.
  pc_table_safe = tf.maximum(pc_table, 0)                        # silence → 0 bin (will be masked out below)
  active_mask = tf.cast(pc_table >= 0, tf.float32)               # [45]
  pc_onehot = tf.one_hot(pc_table_safe, 12, dtype=tf.float32) * \
              active_mask[:, tf.newaxis]                         # [45, 12]
  # pitch_probs [B, T, 45] @ pc_onehot [45, 12] → [B, T, 12]
  pc_probs = tf.einsum('btk,kc->btc', pitch_probs, pc_onehot)
  pc_frame = tf.argmax(pc_probs, axis=-1, output_type=tf.int32)  # [B, T] in 0..11

  # ── Onset segmentation ──────────────────────────────────────────────
  # For each frame, find segment_id = cumsum(onset_peak[<=i]). Then for each
  # segment compute the MODAL pc on non-silence frames in that segment, and
  # broadcast back per-frame.
  if onset_logit_head is not None:
    onset_prob = tf.sigmoid(onset_logit_head)                    # [B, T, 1]
    is_onset = onset_prob > onset_threshold                      # [B, T, 1]
    is_onset_flat = tf.squeeze(is_onset, axis=-1)                # [B, T]

    # segment_id[b, i] = number of onset peaks at indices 0..i, minus 1 if no
    # onset yet (we use cumsum-1 clamped to 0).
    onset_int = tf.cast(is_onset_flat, tf.int32)
    seg_id = tf.cumsum(onset_int, axis=-1) - 1                   # [B, T]
    seg_id = tf.maximum(seg_id, 0)

    pc_segment = _segment_modal_pc(
        pc_frame=pc_frame,
        seg_id=seg_id,
        is_silence=tf.squeeze(is_silence, axis=-1),
    )                                                            # [B, T] in 0..11
  else:
    # No onset info: use per-frame pc directly. Smoother results require
    # the onset path; this is a graceful fallback.
    pc_segment = pc_frame                                        # [B, T]

  # ── Apply pitch-class snap in CREPE's octave ────────────────────────
  f0_c_squeezed = tf.squeeze(f0_c, axis=-1)                      # [B, T]
  midi_crepe = 12.0 * tf.math.log(
      tf.maximum(f0_c_squeezed, 1e-6) / 440.0) / tf.math.log(2.0) + 69.0
  midi_crepe_int = tf.cast(tf.round(midi_crepe), tf.int32)       # nearest MIDI
  # Naive snap: same octave as CREPE, set pc to pc_segment.
  octave = midi_crepe_int // 12                                  # floor div
  midi_fused = octave * 12 + pc_segment                          # int32 [B, T]

  # Octave-correction: if |midi_fused - midi_crepe_int| > 6 we picked the
  # wrong octave. Shift by 12 toward CREPE.
  delta = midi_fused - midi_crepe_int
  midi_fused = tf.where(delta > 6, midi_fused - 12, midi_fused)
  midi_fused = tf.where(delta < -6, midi_fused + 12, midi_fused)

  # Convert back to Hz.
  midi_fused_f = tf.cast(midi_fused, tf.float32)
  f0_snap = 440.0 * tf.pow(2.0, (midi_fused_f - 69.0) / 12.0)    # [B, T]
  f0_snap = f0_snap[..., tf.newaxis]                              # [B, T, 1]

  # ── Silence: bypass fusion, keep CREPE ──────────────────────────────
  f0_fused = tf.where(is_silence, f0_c, f0_snap)

  if squeeze_back:
    f0_fused = tf.squeeze(f0_fused, axis=-1)
  return f0_fused


def _segment_modal_pc(pc_frame: tf.Tensor,
                      seg_id: tf.Tensor,
                      is_silence: tf.Tensor) -> tf.Tensor:
  """Compute the modal pitch class per segment, then broadcast per-frame.

  Args:
    pc_frame:   [B, T] int32 in 0..11
    seg_id:     [B, T] int32 segment id (one int per onset peak)
    is_silence: [B, T] bool — silence frames are excluded from the mode

  Returns:
    pc_per_frame: [B, T] int32 — the modal pc of the segment each frame
      belongs to. Silence frames keep pc_frame's value (irrelevant since
      they'll be bypassed by the silence guard upstream).
  
  Vectorisation: build a [B, n_seg, 12] tensor counting (pc, segment)
  co-occurrences, then argmax over the pc axis.
  """
  B = tf.shape(pc_frame)[0]
  T = tf.shape(pc_frame)[1]
  # Use a fixed upper bound for number of segments: at most T per batch.
  n_seg = T
  
  # Encode each (b, t) frame as (segment_id, pitch_class), masked by
  # non-silence. Use scatter_nd to accumulate counts.
  active = tf.cast(tf.logical_not(is_silence), tf.float32)       # [B, T]
  pc_onehot = tf.one_hot(pc_frame, 12, dtype=tf.float32)         # [B, T, 12]
  pc_weighted = pc_onehot * active[..., tf.newaxis]              # [B, T, 12]

  # Build per-batch (segment, pc) counts via segment_sum.
  # tf.math.unsorted_segment_sum needs a flat segment id; we work per-batch.
  def _per_batch(pc_w, seg):
    # pc_w:  [T, 12]
    # seg:   [T]
    # Returns [n_seg, 12] counts.
    return tf.math.unsorted_segment_sum(pc_w, seg, num_segments=n_seg)
  
  # map_fn over batch dim.
  seg_counts = tf.map_fn(
      lambda x: _per_batch(x[0], x[1]),
      (pc_weighted, seg_id),
      fn_output_signature=tf.float32)                            # [B, n_seg, 12]
  
  # Modal pc per segment.
  modal_pc_seg = tf.argmax(seg_counts, axis=-1, output_type=tf.int32)  # [B, n_seg]
  
  # Gather back per-frame.
  pc_per_frame = tf.gather(modal_pc_seg, seg_id, batch_dims=1)   # [B, T]
  
  # If the segment had ZERO non-silence frames (= an entire silent segment),
  # argmax returns 0 by default — irrelevant since silence guard bypasses.
  # If a frame in a non-silent segment was itself silence, we still apply the
  # segment's modal pc — innocuous because silence_guard will replace this
  # frame with CREPE.
  return pc_per_frame


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_bce(logits, targets, pos_weight, gamma=0.0):
  pw = tf.constant(pos_weight, dtype=tf.float32)
  per_elem = tf.nn.weighted_cross_entropy_with_logits(
      labels=targets, logits=logits, pos_weight=pw)
  if gamma > 0.0:
    p = tf.sigmoid(logits)
    p_t = targets * p + (1.0 - targets) * (1.0 - p)
    per_elem = tf.pow(1.0 - p_t, gamma) * per_elem
  return tf.reduce_mean(per_elem)


def _masked_mse(predictions, targets, mask):
  n = tf.reduce_sum(mask)
  sq_err = tf.square(predictions - targets) * mask
  return tf.cond(n > 0.0,
                 lambda: tf.reduce_sum(sq_err) / n,
                 lambda: tf.constant(0.0))


def _masked_ce(logits, class_targets, mask):
  ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=class_targets, logits=logits)
  ce = ce * mask
  n = tf.reduce_sum(mask)
  return tf.cond(n > 0.0,
                 lambda: tf.reduce_sum(ce) / n,
                 lambda: tf.constant(0.0))


# ─────────────────────────────────────────────────────────────────────────────
# MIDIHeadLoss
# ─────────────────────────────────────────────────────────────────────────────

@gin.register
class MIDIHeadLoss:
  """Computes MIDI head losses.

  Compatibile con Model._update_losses_dict via get_losses_dict().

  Sub-frame normalization: onset_offset_samples è in raw sample
  ∈ [-441, +441] a 44.1 kHz / 50 Hz. HALF_FRAME_SAMPLES = 441.
  """

  HALF_FRAME_SAMPLES = 441.0  # = sample_rate / (2 * frame_rate)

  def __init__(self,
               lambda_onset: float = 1.0,
               lambda_subframe: float = 0.3,
               lambda_silence: float = 1.5,
               lambda_pitch: float = 1.0,
               lambda_velocity: float = 0.3,
               lambda_ks: float = 0.5,
               onset_pos_weight: float = 10.0,
               silence_pos_weight: float = 2.0,
               ks_pos_weight: float = 20.0,
               onset_focal_gamma: float = 1.5,
               name: str = 'midi_head_loss'):
    self.name = name
    self._l_on = float(lambda_onset)
    self._l_sub = float(lambda_subframe)
    self._l_sil = float(lambda_silence)
    self._l_pit = float(lambda_pitch)
    self._l_vel = float(lambda_velocity)
    self._l_ks = float(lambda_ks)
    self._on_pw = float(onset_pos_weight)
    self._sil_pw = float(silence_pos_weight)
    self._ks_pw = float(ks_pos_weight)
    self._on_gam = float(onset_focal_gamma)

  def _expand(self, t):
    return t[..., tf.newaxis] if len(t.shape) == 2 else t

  def get_losses_dict(self, head_outputs: dict, batch: dict) -> dict:
    """All MIDI head losses as named scalars."""
    # Onset / subframe.
    on_mask = self._expand(tf.cast(batch['onset_mask'], tf.float32))
    on_sub_raw = self._expand(tf.cast(
        batch['onset_offset_samples'], tf.float32))
    on_sub_norm = tf.clip_by_value(
        on_sub_raw / self.HALF_FRAME_SAMPLES, -1.0, 1.0)
    l_onset = _weighted_bce(
        head_outputs['onset_logit'], on_mask, self._on_pw, self._on_gam)
    l_sub = _masked_mse(
        head_outputs['onset_subframe'], on_sub_norm, on_mask)

    # Silence.
    sil_mask = self._expand(tf.cast(batch['silence_mask'], tf.float32))
    l_silence = _weighted_bce(
        head_outputs['silence_logit'], sil_mask, self._sil_pw, 0.0)

    # Pitch (supervisione su TUTTI i frame, classe 0 = silenzio).
    note_midi = tf.cast(batch['active_note_midi'], tf.int32)
    pitch_class = tf.where(
        note_midi == 0,
        tf.zeros_like(note_midi),
        note_midi - (MIDIHead.MIDI_PITCH_MIN - 1))
    pitch_class = tf.clip_by_value(
        pitch_class, 0, MIDIHead.N_PITCH_CLASSES - 1)
    pitch_full_mask = tf.ones_like(note_midi, dtype=tf.float32)
    l_pitch = _masked_ce(
        head_outputs['pitch_logits'], pitch_class, pitch_full_mask)

    # Velocity (solo nei frame onset).
    vel_tgt = self._expand(tf.cast(batch['onset_velocity'], tf.float32))
    l_vel = _masked_mse(
        head_outputs['velocity'], vel_tgt, on_mask)

    # KS (multi-label BCE su 12 bit).
    # KS_ACTIVE_MASK azzera bit 3 (KS_NOT_USED, midi_id 12, trigger null):
    # non porta informazione musicale e allenerebbe il modello su etichette
    # sempre-0, sprecando capacità e distorcendo il gradient con pos_weight=20.
    ks_int = tf.cast(batch['active_ks_bits'], tf.int32)
    bits = [tf.cast(
        tf.bitwise.right_shift(ks_int, i) & 1, tf.float32)
        for i in range(MIDIHead.N_KS)]
    ks_mask = tf.constant(MIDIHead.KS_ACTIVE_MASK, dtype=tf.float32)
    ks_targets = tf.stack(bits, axis=-1) * ks_mask
    l_ks = _weighted_bce(
        head_outputs['ks_logits'] * ks_mask, ks_targets, self._ks_pw, 0.0)

    return {
        'midi/onset':     self._l_on  * l_onset,
        'midi/subframe':  self._l_sub * l_sub,
        'midi/silence':   self._l_sil * l_silence,
        'midi/pitch':     self._l_pit * l_pitch,
        'midi/velocity':  self._l_vel * l_vel,
        'midi/ks':        self._l_ks  * l_ks,
    }
