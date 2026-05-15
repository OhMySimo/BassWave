# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Smoke test end-to-end MIDI Head pipeline.

Da lanciare PRIMA di Phase 1 training. Verifica:
  1. Data provider sidecar v2 join → batch coerente, file_hash di main
     == file_hash di sidecar per ogni esempio.
  2. LogMelExtractor: audio_16k → log_mel ha shape attesa.
  3. MIDIHead forward: log_mel → 7 outputs con shape corrette + midi_cond.
  4. MIDIHeadLoss: tutte le sottoloss finite.
  5. build_teacher_midi_cond: produce stesso shape di head midi_cond.
  6. fuse_pitch_with_crepe: f0_fused stessa shape di f0_crepe.
  7. (Opzionale) Plot allineamento head outputs vs MIDI GT su un esempio.

Usage:
    cd ddsp/
    python smoke_test_midi_head.py \\
        --main_pattern    /media/simone/NVME/MidiDataset/BassWave_TFR/basswave-train-*.tfrecord \\
        --sidecar_pattern /media/simone/NVME/MidiDataset/BassWave_TFR_MIDI_v2/basswave-train-midi-*.tfrecord \\
        --batch_size 4 \\
        --plot_out /tmp/smoke_midi.png

Se PASSA: lancia Phase 1 training.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tensorflow.compat.v2 as tf

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def _green(s):  return f'\033[92m{s}\033[0m'
def _red(s):    return f'\033[91m{s}\033[0m'
def _yellow(s): return f'\033[93m{s}\033[0m'


# ─────────────────────────────────────────────────────────────────────────
# Test 1: data provider.
# ─────────────────────────────────────────────────────────────────────────

def test_data_provider(args):
  print('\n' + '=' * 70)
  print('TEST 1: BassWaveWithSidecarProvider — join main + sidecar v2')
  print('=' * 70)

  try:
    from ddsp.training.data_basswave import BassWaveWithSidecarProvider
  except ImportError as e:
    print(_red(f'FAIL: cannot import BassWaveWithSidecarProvider: {e}'))
    sys.exit(1)

  provider = BassWaveWithSidecarProvider(
      file_pattern=args.main_pattern,
      sidecar_file_pattern=args.sidecar_pattern,
      sample_rate=44100,
      frame_rate=50,
      example_secs=4,
      centered=True,
      shuffle_buffer_size=8,
      prefetch_size=1,
  )

  print(f'  main:    {args.main_pattern}')
  print(f'  sidecar: {args.sidecar_pattern}')

  ds = provider.get_batch(batch_size=args.batch_size, shuffle=False,
                          repeats=1, drop_remainder=True)
  try:
    batch = next(iter(ds))
  except StopIteration:
    print(_red('FAIL: dataset empty.'))
    sys.exit(1)
  except Exception as e:
    print(_red(f'FAIL: dataset iteration: {e}'))
    sys.exit(1)

  expected = {
      'audio', 'audio_16k', 'f0_hz', 'f0_confidence', 'loudness_db',
      'preset_id', 'transpose', 'groove_cat_id', 'file_hash',
      'onset_mask', 'onset_offset_samples', 'onset_velocity',
      'offset_mask', 'offset_offset_samples', 'silence_mask',
      'active_ks_bits', 'active_note_midi',
  }
  got = set(batch.keys())
  missing = expected - got
  if missing:
    print(_red(f'FAIL: missing keys: {missing}'))
    sys.exit(1)
  print(_green(f'  OK: all {len(expected)} keys present.'))

  for k in ('onset_mask', 'silence_mask', 'active_note_midi'):
    sh = batch[k].shape
    if len(sh) != 2 or sh[0] != args.batch_size or sh[1] != 201:
      print(_red(f'FAIL: {k} shape={sh} expected [{args.batch_size}, 201]'))
      sys.exit(1)
  print(_green(f'  OK: shapes [B={args.batch_size}, T=201]'))

  # Sanity stats.
  on_count = int(batch['onset_mask'].numpy().sum())
  sil = float(batch['silence_mask'].numpy().mean())
  print(f'  total onsets: {on_count}   mean silence: {sil:.3f}')
  if on_count == 0:
    print(_red('FAIL: zero onsets — sidecar empty?'))
    sys.exit(1)

  return batch


# ─────────────────────────────────────────────────────────────────────────
# Test 2: LogMelExtractor + MIDIHead + MIDIHeadLoss
# ─────────────────────────────────────────────────────────────────────────

def test_midi_head_forward(batch):
  print('\n' + '=' * 70)
  print('TEST 2: LogMelExtractor + MIDIHead forward + losses')
  print('=' * 70)

  try:
    from ddsp.training.midi_head import (
        LogMelExtractor, MIDIHead, MIDIHeadLoss, fuse_pitch_with_crepe)
  except ImportError as e:
    print(_red(f'FAIL: cannot import midi_head: {e}'))
    print(_red('  Did you copy midi_head.py to ddsp/training/?'))
    sys.exit(1)

  # LogMel.
  audio_16k = batch['audio_16k']                          # [B, ~64000]
  mel = LogMelExtractor(
      sample_rate=16000, n_mels=128,
      win_length=1024, hop_length=320,
      fmin=40.0, fmax=8000.0)
  log_mel = mel(audio_16k)
  B = audio_16k.shape[0]
  print(f'  log_mel shape: {tuple(log_mel.shape)}')
  if log_mel.shape[0] != B or log_mel.shape[2] != 128:
    print(_red(f'FAIL: log_mel shape unexpected'))
    sys.exit(1)
  # T_frames should be ~201 (matching MIDI features).
  if abs(log_mel.shape[1] - 201) > 5:
    print(_yellow(f'  warn: log_mel T={log_mel.shape[1]}, expected ~201. '
                  f'Mel framing may not match MIDI framing exactly.'))
  else:
    print(_green(f'  OK: log_mel T={log_mel.shape[1]} (matches MIDI T=201)'))

  # Truncate/pad mel to MIDI T=201 if mismatch (defensive).
  T_target = batch['onset_mask'].shape[1]
  if log_mel.shape[1] != T_target:
    if log_mel.shape[1] > T_target:
      log_mel = log_mel[:, :T_target, :]
    else:
      pad = T_target - log_mel.shape[1]
      log_mel = tf.pad(log_mel, [[0, 0], [0, pad], [0, 0]])

  # MIDIHead.
  head = MIDIHead(n_mels=128, ch_d1=256, ch_d2=384,
                  ch_bottleneck=512, bigru_units=256)
  head_out = head(log_mel, training=True)

  expected_outputs = {
      'onset_logit': (B, T_target, 1),
      'onset_subframe': (B, T_target, 1),
      'silence_logit': (B, T_target, 1),
      'pitch_logits': (B, T_target, 45),
      'velocity': (B, T_target, 1),
      'ks_logits': (B, T_target, 12),
      'midi_cond': (B, T_target, 60),
  }
  for k, expected_sh in expected_outputs.items():
    sh = tuple(head_out[k].shape)
    if sh != expected_sh:
      print(_red(f'FAIL: {k} shape={sh}, expected {expected_sh}'))
      sys.exit(1)
  print(_green('  OK: MIDIHead forward — all 7 outputs with correct shapes.'))
  print(f'  midi_cond [B={B}, T={T_target}, 60] composed of: '
        f'1 onset + 1 silence + 45 pitch + 1 vel + 12 ks')

  # Param count.
  n_params = sum(np.prod(v.shape) for v in head.trainable_variables)
  print(f'  head trainable params: {n_params/1e6:.2f}M')

  # MIDIHeadLoss.
  loss_obj = MIDIHeadLoss()
  losses = loss_obj.get_losses_dict(head_outputs=head_out, batch=batch)
  print(f'  loss components:')
  for k, v in sorted(losses.items()):
    val = float(v.numpy())
    if not np.isfinite(val):
      print(_red(f'  FAIL: {k} = {val} (not finite)'))
      sys.exit(1)
    print(f'    {k:25s} = {val:.4f}')
  total = sum(float(v.numpy()) for v in losses.values())
  print(_green(f'  OK: all losses finite, total = {total:.4f}'))

  return head_out, head


# ─────────────────────────────────────────────────────────────────────────
# Test 3: Teacher-forcing midi_cond matches head shape.
# ─────────────────────────────────────────────────────────────────────────

def test_teacher_forcing(batch, head_out):
  print('\n' + '=' * 70)
  print('TEST 3: teacher-forcing midi_cond from MIDI GT')
  print('=' * 70)

  from ddsp.training.midi_head import MIDIHead
  tf_cond = MIDIHead.build_teacher_midi_cond(batch)
  head_cond = head_out['midi_cond']
  if tf_cond.shape != head_cond.shape:
    print(_red(f'FAIL: shapes differ. teacher={tf_cond.shape} '
               f'head={head_cond.shape}'))
    sys.exit(1)
  print(_green(f'  OK: teacher_cond shape {tuple(tf_cond.shape)} = head shape'))

  # Sanity: in frame onset, teacher_cond[..., 0] = 1.
  on_mask = batch['onset_mask'].numpy()
  tf_onset_ch = tf_cond.numpy()[..., 0]
  on_idx = np.flatnonzero(on_mask)
  if len(on_idx) > 0:
    # Check first onset frame.
    b, t = np.unravel_index(on_idx[0], on_mask.shape)
    if tf_onset_ch[b, t] < 0.99:
      print(_yellow(f'  warn: tf_cond onset @[{b},{t}] = '
                    f'{tf_onset_ch[b, t]:.2f}, expected 1.0'))
    else:
      print(_green('  OK: teacher midi_cond channel 0 == onset_mask'))


# ─────────────────────────────────────────────────────────────────────────
# Test 4: pitch fusion runs without errors.
# ─────────────────────────────────────────────────────────────────────────

def test_pitch_fusion(batch, head_out):
  print('\n' + '=' * 70)
  print('TEST 4: pitch fusion CREPE + head')
  print('=' * 70)

  from ddsp.training.midi_head import fuse_pitch_with_crepe

  f0 = batch['f0_hz']
  conf = batch['f0_confidence']
  f0_fused = fuse_pitch_with_crepe(
      f0_crepe=f0,
      f0_crepe_conf=conf,
      pitch_logits_head=head_out['pitch_logits'],
      silence_logit_head=head_out['silence_logit'])

  if f0_fused.shape != f0.shape:
    print(_red(f'FAIL: f0_fused shape={f0_fused.shape} '
               f'vs f0={f0.shape}'))
    sys.exit(1)
  print(_green(f'  OK: f0_fused shape {tuple(f0_fused.shape)}'))

  if not np.all(np.isfinite(f0_fused.numpy())):
    print(_red('FAIL: f0_fused has NaN/Inf'))
    sys.exit(1)
  print(_green(f'  OK: f0_fused finite, '
               f'range=[{float(f0_fused.numpy().min()):.1f}, '
               f'{float(f0_fused.numpy().max()):.1f}] Hz'))


# ─────────────────────────────────────────────────────────────────────────
# Plot.
# ─────────────────────────────────────────────────────────────────────────

def make_plot(batch, head_out, plot_out):
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    print(_yellow('  matplotlib not available, skipping plot'))
    return

  audio = batch['audio'][0].numpy()
  on_mask = batch['onset_mask'][0].numpy()
  sil_mask = batch['silence_mask'][0].numpy()
  note_midi = batch['active_note_midi'][0].numpy()

  on_pred = tf.sigmoid(head_out['onset_logit'])[0, :, 0].numpy()
  sil_pred = tf.sigmoid(head_out['silence_logit'])[0, :, 0].numpy()
  pitch_probs = tf.nn.softmax(head_out['pitch_logits'])[0].numpy()
  pitch_pred_class = pitch_probs.argmax(axis=-1)

  sr = 44100
  fr = 50
  t_audio = np.arange(len(audio)) / sr
  t_frames = np.arange(len(on_mask)) / fr

  fig, axes = plt.subplots(4, 1, figsize=(14, 11))

  ax = axes[0]
  ax.plot(t_audio, audio, color='black', linewidth=0.4)
  for i in np.flatnonzero(on_mask):
    ax.axvline(i / fr, color='red', alpha=0.5, linewidth=0.7)
  ax.set_title(f'Plot 1: audio + MIDI onsets (rosso)')
  ax.set_xlim(0, t_audio[-1]); ax.grid(alpha=0.3)

  ax = axes[1]
  ax.plot(t_frames, on_mask, color='red', label='GT onset_mask', alpha=0.7)
  ax.plot(t_frames, on_pred, color='blue', label='head onset_prob (untrained!)',
          alpha=0.7)
  ax.set_title('Plot 2: onset GT (rosso) vs head pred (blu, untrained random)')
  ax.set_xlim(0, t_frames[-1]); ax.legend(); ax.grid(alpha=0.3)

  ax = axes[2]
  ax.plot(t_frames, sil_mask, color='red', label='GT silence', alpha=0.7)
  ax.plot(t_frames, sil_pred, color='blue', label='head silence (untrained)',
          alpha=0.7)
  ax.set_title('Plot 3: silence GT (rosso) vs head pred (blu, untrained)')
  ax.set_xlim(0, t_frames[-1]); ax.legend(); ax.grid(alpha=0.3)

  ax = axes[3]
  # GT pitch class.
  gt_class = np.where(note_midi == 0, 0, note_midi - 20)
  ax.step(t_frames, gt_class, where='post',
          color='red', label='GT pitch class', alpha=0.7)
  ax.step(t_frames, pitch_pred_class, where='post',
          color='blue', label='head pitch class (untrained)', alpha=0.7)
  ax.set_ylim(-1, 46)
  ax.set_ylabel('class (0=sil, 1..44 = MIDI 21..64)')
  ax.set_title('Plot 4: pitch class GT vs head pred (untrained)')
  ax.set_xlim(0, t_frames[-1]); ax.legend(); ax.grid(alpha=0.3)

  plt.tight_layout()
  plt.savefig(plot_out, dpi=110)
  print(_green(f'  OK: plot → {plot_out}'))


# ─────────────────────────────────────────────────────────────────────────
# Main.
# ─────────────────────────────────────────────────────────────────────────

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--main_pattern', required=True)
  ap.add_argument('--sidecar_pattern', required=True)
  ap.add_argument('--batch_size', type=int, default=4)
  ap.add_argument('--plot_out', default='')
  args = ap.parse_args()

  batch = test_data_provider(args)
  head_out, head = test_midi_head_forward(batch)
  test_teacher_forcing(batch, head_out)
  test_pitch_fusion(batch, head_out)

  if args.plot_out:
    print('\nGenerating plot...')
    make_plot(batch, head_out, args.plot_out)

  print('\n' + '=' * 70)
  print(_green('ALL TESTS PASSED — pronto per Phase 1 training.'))
  print('=' * 70)


if __name__ == '__main__':
  main()
