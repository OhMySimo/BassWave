# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Smoke test per la pipeline TransientHead end-to-end.

Da eseguire PRIMA di lanciare 100k step di training v4. Verifica:
  1. Il join sidecar -> main funziona: dataset legge entrambi i TFRecord
     in modo coerente (file_hash di main == file_hash di sidecar per
     ogni esempio del batch).
  2. Le feature MIDI hanno le statistiche attese (onset count, silence
     fraction, range velocity).
  3. TransientHead forward pass produce output con shape corrette.
  4. TransientLoss forward computa scalari finiti (no NaN/inf).
  5. (Opzionale) Plot di un esempio: audio + onset_mask + silence_mask
     overlay, per verifica visiva join.

Usage:
    cd ddsp/
    python smoke_test_transient_head.py \\
        --main_pattern    /media/simone/NVME/MidiDataset/BassWave_TFR/basswave-train-*.tfrecord \\
        --sidecar_pattern /media/simone/NVME/MidiDataset/BassWave_TFR_MIDI/basswave-train-midi-*.tfrecord \\
        --batch_size 4 \\
        --plot_out /tmp/smoke_align.png

Se PASSA: il join e le shape sono corrette, puoi lanciare il training.
Se FALLISCE: il messaggio di errore dice esattamente cosa non va.

ATTENZIONE: questo script va lanciato DOPO aver applicato il patch
BassWaveWithSidecarProvider a data_basswave.py. Senza il patch, l'import
sotto fallisce subito.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tensorflow.compat.v2 as tf

# Silenzia TF info.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def _green(s):  return f'\033[92m{s}\033[0m'
def _red(s):    return f'\033[91m{s}\033[0m'
def _yellow(s): return f'\033[93m{s}\033[0m'


# -----------------------------------------------------------------------------
# Test 1: data provider join works.
# -----------------------------------------------------------------------------


def test_data_provider(args):
  print('\n' + '=' * 70)
  print('TEST 1: BassWaveWithSidecarProvider — join main + sidecar')
  print('=' * 70)

  try:
    from ddsp.training.data_basswave import BassWaveWithSidecarProvider
  except ImportError as e:
    print(_red('FAIL: cannot import BassWaveWithSidecarProvider.'))
    print(_red(f'  Did you patch data_basswave.py? Error: {e}'))
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

  print(f'  main pattern    : {args.main_pattern}')
  print(f'  sidecar pattern : {args.sidecar_pattern}')

  ds = provider.get_batch(batch_size=args.batch_size, shuffle=False,
                          repeats=1, drop_remainder=True)
  it = iter(ds)
  try:
    batch = next(it)
  except StopIteration:
    print(_red('FAIL: dataset is empty. Check TFR patterns.'))
    sys.exit(1)
  except Exception as e:
    print(_red(f'FAIL: dataset iteration error: {e}'))
    sys.exit(1)

  print(_green('  OK: batch fetched.'))
  expected_keys = {
      'audio', 'audio_16k', 'f0_hz', 'f0_confidence', 'loudness_db',
      'preset_id', 'transpose', 'groove_cat_id', 'file_hash',
      'onset_mask', 'onset_offset_samples', 'onset_velocity',
      'offset_mask', 'offset_offset_samples', 'silence_mask',
      'active_ks_bits', 'active_note_midi',
  }
  got = set(batch.keys())
  missing = expected_keys - got
  extra = got - expected_keys
  if missing:
    print(_red(f'FAIL: missing keys: {missing}'))
    sys.exit(1)
  if extra:
    print(_yellow(f'  warn: extra keys present (harmless): {extra}'))
  print(_green(f'  OK: all {len(expected_keys)} expected keys present.'))

  # Verifica shape.
  for k in ('onset_mask', 'silence_mask', 'onset_velocity'):
    sh = batch[k].shape
    if len(sh) != 2 or sh[0] != args.batch_size or sh[1] != 201:
      print(_red(f'FAIL: {k} shape={sh}, expected [{args.batch_size}, 201]'))
      sys.exit(1)
  print(_green('  OK: shapes are [B=%d, T=201]' % args.batch_size))

  return batch, provider


# -----------------------------------------------------------------------------
# Test 2: MIDI feature stats reasonable.
# -----------------------------------------------------------------------------


def test_feature_stats(batch):
  print('\n' + '=' * 70)
  print('TEST 2: MIDI feature stats')
  print('=' * 70)

  onset_mask  = batch['onset_mask'].numpy()
  silence     = batch['silence_mask'].numpy()
  vel         = batch['onset_velocity'].numpy()
  sub_samples = batch['onset_offset_samples'].numpy()
  ks_bits     = batch['active_ks_bits'].numpy()

  B, T = onset_mask.shape
  total_onsets = int(onset_mask.sum())
  mean_sil     = float(silence.mean())
  onset_rate   = total_onsets / float(B * T)

  print(f'  batch size              : {B}')
  print(f'  T_frames                : {T}')
  print(f'  total onsets            : {total_onsets}')
  print(f'  onset rate per frame    : {onset_rate * 100:.2f}%')
  print(f'  mean silence fraction   : {mean_sil:.3f}')

  # Sanity ranges.
  if total_onsets == 0:
    print(_red('FAIL: zero onsets in batch — sidecar empty or misaligned.'))
    sys.exit(1)
  if onset_rate > 0.3:
    print(_yellow(f'  warn: onset rate {onset_rate*100:.1f}% is unusually '
                  f'high. Expected 1-15%.'))
  if mean_sil < 0.05 or mean_sil > 0.95:
    print(_yellow(f'  warn: silence fraction {mean_sil:.3f} is extreme. '
                  f'Expected ~0.2-0.6 for bass.'))

  # Onset velocity range.
  on_idx = np.flatnonzero(onset_mask.flatten())
  if len(on_idx) > 0:
    vel_flat = vel.flatten()[on_idx]
    print(f'  velocity range          : '
          f'[{vel_flat.min():.3f}, {vel_flat.max():.3f}]  '
          f'mean={vel_flat.mean():.3f}')
    if vel_flat.min() < 0 or vel_flat.max() > 1.0:
      print(_red('FAIL: velocity out of [0, 1].'))
      sys.exit(1)

    # Sub-sample residue range — should be in [-441, +441] at 44100/50.
    sub_flat = sub_samples.flatten()[on_idx]
    print(f'  sub-sample residue      : '
          f'[{sub_flat.min():.1f}, {sub_flat.max():.1f}]  '
          f'mean={sub_flat.mean():.1f}')
    if sub_flat.min() < -442 or sub_flat.max() > 442:
      print(_red('FAIL: sub-sample residue out of [-441, +441].'))
      sys.exit(1)

  # KS bits decoding.
  ks_active_frames = int((ks_bits > 0).sum())
  print(f'  KS-active frames        : {ks_active_frames} '
        f'({100 * ks_active_frames / (B * T):.2f}%)')

  print(_green('  OK: feature stats look reasonable.'))


# -----------------------------------------------------------------------------
# Test 3 + 4: TransientHead + TransientLoss forward.
# -----------------------------------------------------------------------------


def test_head_and_loss(batch):
  print('\n' + '=' * 70)
  print('TEST 3+4: TransientHead forward + TransientLoss')
  print('=' * 70)

  try:
    from ddsp.training.transient_head import TransientHead, TransientLoss
  except ImportError as e:
    print(_red(f'FAIL: cannot import transient_head. Error: {e}'))
    print(_red('  Did you copy transient_head.py to ddsp/training/?'))
    sys.exit(1)

  z_dim = 48
  T = batch['onset_mask'].shape[1]
  B = batch['onset_mask'].shape[0]

  # Fake z (no encoder needed for head smoke test).
  z = tf.random.normal([B, T, z_dim])
  head = TransientHead(z_dim=z_dim, hidden_channels=128, n_layers=3,
                       sample_rate=44100, frame_rate=50)

  head_out = head(z, training=True)
  print(f'  head output keys        : {sorted(head_out.keys())}')
  for k, v in head_out.items():
    if v.shape != (B, T, 1):
      print(_red(f'FAIL: {k} shape={v.shape}, expected ({B}, {T}, 1)'))
      sys.exit(1)
  print(_green('  OK: TransientHead forward — 6 outputs each [B, T, 1].'))

  # Test loss.
  loss_obj = TransientLoss()
  losses = loss_obj.get_losses_dict(
      outputs=head_out, batch=batch, head=head)

  print(f'  loss keys              : {sorted(losses.keys())}')
  for k, v in losses.items():
    val = float(v.numpy())
    if not np.isfinite(val):
      print(_red(f'FAIL: loss {k} = {val} (not finite)'))
      sys.exit(1)
    print(f'    {k:30s} = {val:.4f}')
  print(_green('  OK: TransientLoss forward — all losses finite.'))

  return head, head_out, losses


# -----------------------------------------------------------------------------
# Plot (optional).
# -----------------------------------------------------------------------------


def make_plot(batch, plot_out):
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    print(_yellow('  matplotlib not available — skipping plot.'))
    return

  audio    = batch['audio'][0].numpy()
  onset    = batch['onset_mask'][0].numpy()
  silence  = batch['silence_mask'][0].numpy()
  pitch    = batch['active_note_midi'][0].numpy()

  sr = 44100
  fr = 50
  t_audio = np.arange(len(audio)) / sr
  t_frames = np.arange(len(onset)) / fr

  fig, axes = plt.subplots(3, 1, figsize=(14, 9))

  ax = axes[0]
  ax.plot(t_audio, audio, color='black', linewidth=0.5)
  onset_idx = np.flatnonzero(onset)
  for i in onset_idx:
    ax.axvline(i / fr, color='red', alpha=0.6, linewidth=0.7)
  ax.set_title(f'Plot 1: audio + onsets (rosso). {len(onset_idx)} onsets.')
  ax.set_xlim(0, t_audio[-1])
  ax.grid(alpha=0.3)

  ax = axes[1]
  ax.plot(t_audio, audio, color='black', linewidth=0.5)
  silence_bool = silence > 0.5
  if silence_bool.any():
    diff = np.diff(silence_bool.astype(int))
    starts = np.flatnonzero(diff == 1) + 1
    ends = np.flatnonzero(diff == -1) + 1
    if silence_bool[0]:
      starts = np.r_[0, starts]
    if silence_bool[-1]:
      ends = np.r_[ends, len(silence_bool)]
    for s, e in zip(starts, ends):
      ax.axvspan(s / fr, e / fr, color='gray', alpha=0.3)
  ax.set_title(
      f'Plot 2: audio + silence_mask (grigio). silence_frac={silence.mean():.3f}')
  ax.set_xlim(0, t_audio[-1])
  ax.grid(alpha=0.3)

  ax = axes[2]
  ax.step(t_frames, pitch, where='post', color='navy')
  ax.set_ylabel('MIDI pitch')
  ax.set_xlim(0, t_audio[-1])
  ax.set_title('Plot 3: active_note_midi (0=silence).')
  ax.grid(alpha=0.3)

  plt.tight_layout()
  plt.savefig(plot_out, dpi=110)
  print(_green(f'  OK: plot saved to {plot_out}'))


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--main_pattern',    required=True,
                  help='Glob TFR principali (train).')
  ap.add_argument('--sidecar_pattern', required=True,
                  help='Glob TFR sidecar MIDI (train).')
  ap.add_argument('--batch_size',      type=int, default=4)
  ap.add_argument('--plot_out',        default='',
                  help='Path PNG di verifica visiva (vuoto=disabilita).')
  args = ap.parse_args()

  batch, _ = test_data_provider(args)
  test_feature_stats(batch)
  test_head_and_loss(batch)

  if args.plot_out:
    print('\nGenerating verification plot...')
    make_plot(batch, args.plot_out)

  print('\n' + '=' * 70)
  print(_green('ALL TESTS PASSED — pronto per training v4.'))
  print('=' * 70)


if __name__ == '__main__':
  main()
