# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Test allineamento MIDI/FLAC per midi_features.py.

Prende una coppia (MIDI, FLAC) reale, runna il parser, genera le feature per
finestra, e produce un PNG di verifica visiva con:

  Plot 1 - Waveform 44.1 kHz dell'intero FLAC con linee verticali rosse
           sui MIDI onset e linee blu su MIDI offset. Se l'allineamento
           e' corretto, le linee rosse devono cadere ESATTAMENTE sui
           transient. Key switch (pitch 9-20) mostrati in verde tratteggiato.

  Plot 2 - Stessa waveform con sovrapposizione del silence_mask (zona
           ombreggiata = MIDI dice "silence").

  Plot 3 - Per la prima finestra: onset_mask come stem plot (frame
           indices) accanto a un envelope RMS dell'audio della finestra.
           Annotazioni: frame_idx / sub-frame offset / velocity.

  Plot 4 - active_ks_bits per la prima finestra (bitmask decodificata per
           KS index) + active_note_midi. Mostrato solo se --keymap e' fornito.

Usage:
    python test_midi_alignment.py \\
        --midi /media/simone/NVME/MidiDataset/MIDI_AUG/.../Variation_01_T-3.mid \\
        --flac /media/simone/NVME/MidiDataset/FLAC_AUG/.../Variation_01__Tp-3__Pk*.flac \\
        --output /tmp/midi_test.png \\
        --keymap /media/simone/NVME/MidiDataset/bass_midi_keymap.json \\
        --override_bpm 120 \\
        --offset_ms 6.35
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from midi_features import (
    BassMidiKeymap, parse_midi, windowed_features_for_file,
    n_frames_for_window)

# KS token names in order (index 0..11 = MIDI 9..20).
_KS_NAMES = [
    'POLYPHONY', 'HARMONIC', 'SLAP_POP', 'NOT_USED',
    'MID_FIN_DW', 'REPEATER', 'LOUD_MUTE', 'GHOST_NOTE',
    'SLIDE_DOWN', 'SLIDE', 'SLIDE_UP', 'LEGATO',
]


# -----------------------------------------------------------------------------
# Audio helpers.
# -----------------------------------------------------------------------------


def rms_envelope(audio: np.ndarray, sr: int, window_ms: float = 20.0,
                 hop_ms: float = 20.0):
  win = max(1, int(round(window_ms * 1e-3 * sr)))
  hop = max(1, int(round(hop_ms * 1e-3 * sr)))
  pad = win // 2
  a = np.pad(audio.astype(np.float64), (pad, pad))
  out, ts = [], []
  i = 0
  while i + win <= len(a):
    seg = a[i:i+win]
    out.append(np.sqrt(np.mean(seg * seg) + 1e-12))
    ts.append((i - pad + win / 2) / sr)
    i += hop
  return np.array(ts), np.array(out)


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--midi', required=True)
  ap.add_argument('--flac', required=True)
  ap.add_argument('--output', default='./midi_test.png')
  ap.add_argument('--keymap', default='',
                  help='Path bass_midi_keymap.json. Se fornito, separa KS '
                       'da note, abilita Plot 4, mostra KS in verde.')
  ap.add_argument('--example_secs', type=float, default=4.0)
  ap.add_argument('--hop_secs',     type=float, default=2.0)
  ap.add_argument('--frame_rate',   type=int,   default=50)
  ap.add_argument('--onset_blur',   type=int,   default=0)
  ap.add_argument('--show_window',  type=int,   default=0)
  ap.add_argument('--override_bpm', type=float, default=None)
  ap.add_argument('--offset_ms',    type=float, default=0.0,
                  help='Offset costante audio-MIDI in ms.')
  args = ap.parse_args()

  # ── Carica keymap (opzionale) ──────────────────────────────────────────
  keymap = None
  if args.keymap and os.path.exists(args.keymap):
    keymap = BassMidiKeymap(args.keymap)
    print(f'[keymap] loaded: {args.keymap}')
  else:
    print('[keymap] not provided — KS will appear as regular notes.')

  # ── Load audio ────────────────────────────────────────────────────────
  audio, sr = sf.read(args.flac, dtype='float32', always_2d=False)
  if audio.ndim > 1:
    audio = audio.mean(axis=1)
  dur = len(audio) / sr
  print(f'[flac] {args.flac}')
  print(f'  sr={sr}  duration={dur:.3f}s  samples={len(audio)}')

  # ── Parse MIDI ────────────────────────────────────────────────────────
  notes, ks_events = parse_midi(
      args.midi,
      override_bpm=args.override_bpm,
      keymap=keymap)

  if args.offset_ms != 0.0:
    offset_s = args.offset_ms / 1000.0
    for n in notes:
      n.t_on  += offset_s
      n.t_off += offset_s
    for ev in ks_events:
      ev.t_on  += offset_s
      ev.t_off += offset_s

  print(f'[midi] {args.midi}')
  print(f'  note events    = {len(notes)}  (pitch 21-64)')
  print(f'  KS events      = {len(ks_events)}  (pitch 9-20, separati)')
  if notes:
    print(f'  first note : t_on={notes[0].t_on:.3f}s '
          f'pitch={notes[0].pitch} token={notes[0].token_id}')
    print(f'  last note  : t_off={notes[-1].t_off:.3f}s')
  if ks_events:
    print(f'  first KS   : t_on={ks_events[0].t_on:.3f}s '
          f'token={ks_events[0].token_id} trigger={ks_events[0].trigger}')

  # ── Generate per-window features ──────────────────────────────────────
  wins = windowed_features_for_file(
      notes, ks_events=ks_events, file_duration_s=dur,
      example_secs=args.example_secs, hop_secs=args.hop_secs,
      frame_rate=args.frame_rate, centered=True, onset_blur=args.onset_blur)
  T = n_frames_for_window(args.example_secs, args.frame_rate, centered=True)
  print(f'[features] windows={len(wins)}  T_frames_per_window={T}')

  onset_total  = sum(int(w['onset_mask'].sum())  for w in wins)
  offset_total = sum(int(w['offset_mask'].sum()) for w in wins)
  silence_frac = np.mean([w['silence_mask'].mean() for w in wins])
  any_ks       = any(w['active_ks_bits'].max() > 0 for w in wins)
  print(f'  total onsets:   {onset_total}')
  print(f'  total offsets:  {offset_total}')
  print(f'  mean silence:   {silence_frac:.3f}')
  print(f'  KS active:      {"yes" if any_ks else "no (keymap not loaded)"}')
  print(f'  (raw MIDI notes: {len(notes)}, raw KS: {len(ks_events)})')

  # ── Layout: 3 or 4 plots ──────────────────────────────────────────────
  n_plots = 4 if keymap else 3
  fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots),
                           gridspec_kw={'height_ratios': [1] * n_plots})
  if n_plots == 3:
    ax1, ax2, ax3 = axes
    ax4 = None
  else:
    ax1, ax2, ax3, ax4 = axes

  t_audio = np.arange(len(audio)) / sr

  # ── Plot 1: waveform + onset/offset + KS markers ─────────────────────
  ax1.plot(t_audio, audio, color='black', linewidth=0.4, alpha=0.7)
  for n in notes:
    ax1.axvline(n.t_on,  color='red',  linewidth=0.5, alpha=0.7)
    ax1.axvline(n.t_off, color='blue', linewidth=0.5, alpha=0.4)
  for ev in ks_events:
    ax1.axvline(ev.t_on, color='green', linewidth=0.6,
                alpha=0.8, linestyle='--')
  ax1.set_xlim(0, dur)
  ax1.set_ylabel('audio')
  ax1.set_title(
      f'Plot 1 — Waveform + note onsets (rosso) / offsets (blu) '
      f'/ KS (verde tratt.)  '
      f'[{len(notes)} note, {len(ks_events)} KS]')
  ax1.grid(True, alpha=0.3)

  # ── Plot 2: waveform + silence-mask ───────────────────────────────────
  total_frames_global = int(np.ceil(dur * args.frame_rate)) + 1
  silence_global = np.ones(total_frames_global, dtype=np.float32)
  count_global   = np.zeros(total_frames_global, dtype=np.int32)
  for w_idx, w in enumerate(wins):
    t0 = w_idx * args.hop_secs
    start_frame = int(round(t0 * args.frame_rate))
    end_frame = start_frame + T
    end_frame_clip = min(end_frame, total_frames_global)
    seg_len = end_frame_clip - start_frame
    silence_global[start_frame:end_frame_clip] = np.minimum(
        silence_global[start_frame:end_frame_clip],
        w['silence_mask'][:seg_len])
    count_global[start_frame:end_frame_clip] += 1
  silence_global[count_global == 0] = 0.0

  ax2.plot(t_audio, audio, color='black', linewidth=0.4, alpha=0.7)
  t_frames = np.arange(total_frames_global) / args.frame_rate
  in_silence = silence_global > 0.5
  if np.any(in_silence):
    diff = np.diff(in_silence.astype(np.int32))
    starts = np.flatnonzero(diff == 1) + 1
    ends   = np.flatnonzero(diff == -1) + 1
    if in_silence[0]:
      starts = np.r_[0, starts]
    if in_silence[-1]:
      ends = np.r_[ends, len(in_silence)]
    for s, e in zip(starts, ends):
      ax2.axvspan(t_frames[s] if s < len(t_frames) else dur,
                  t_frames[e-1] if e-1 < len(t_frames) else dur,
                  color='gray', alpha=0.3)
  ax2.set_xlim(0, dur)
  ax2.set_ylabel('audio')
  ax2.set_title(
      f'Plot 2 — Waveform + silence-mask (grigio). '
      f'Silence globale: {silence_global[count_global > 0].mean():.3f}')
  ax2.grid(True, alpha=0.3)

  # ── Plot 3: zoom su una finestra, onset stem + RMS ────────────────────
  iw = max(0, min(args.show_window, len(wins) - 1))
  w = wins[iw]
  t0 = iw * args.hop_secs
  t_end = t0 + args.example_secs
  s_start = int(round(t0 * sr))
  s_end   = int(round(t_end * sr))
  audio_win = audio[s_start:s_end]
  t_win = np.arange(len(audio_win)) / sr + t0
  ts_env, env = rms_envelope(audio_win, sr, window_ms=20.0, hop_ms=20.0)
  ts_env += t0

  ax3.plot(t_win, audio_win, color='black', linewidth=0.4,
           alpha=0.5, label='audio')
  ax3.plot(ts_env, env / (env.max() + 1e-6) * 0.9, color='green',
           linewidth=1.2, label='RMS env (norm)')
  onset_idx = np.flatnonzero(w['onset_mask'])
  for f_idx in onset_idx:
    t_frame = t0 + f_idx / args.frame_rate
    sub_samples = float(w['onset_offset_samples'][f_idx])
    vel = float(w['onset_velocity'][f_idx])
    color = (1.0, 1.0 - 0.8 * vel, 1.0 - 0.8 * vel)
    ax3.axvline(t_frame, color=color, linewidth=0.8 + 1.2 * vel, alpha=0.85)
    ax3.text(t_frame, 0.95,
             f'{f_idx}\n{sub_samples:+.0f}sa\nv={vel:.2f}',
             rotation=90, fontsize=6.5, color='darkred',
             ha='right', va='top',
             transform=ax3.get_xaxis_transform())
  ax3.set_xlim(t0, t_end)
  ax3.set_xlabel('time (s)')
  ax3.set_ylabel('audio / env')
  ax3.set_title(
      f'Plot 3 — Finestra {iw} [{t0:.2f}-{t_end:.2f}s]. '
      f'Onset frames (rosso) + RMS env (verde).')
  ax3.legend(loc='upper right', fontsize=8)
  ax3.grid(True, alpha=0.3)

  # ── Plot 4: KS bitmask + active note MIDI (solo con keymap) ───────────
  if ax4 is not None:
    frame_t = np.arange(T) / args.frame_rate + t0
    ks_bits = w['active_ks_bits'].astype(int)
    note_midi = w['active_note_midi']

    # Un subplot per ogni KS che e' attivo in questa finestra.
    active_ks_indices = [i for i in range(BassMidiKeymap.N_KS)
                         if np.any(ks_bits & (1 << i))]

    # KS state: stacked lines per bit.
    offset_y = 0
    for i in active_ks_indices:
      bit_active = ((ks_bits & (1 << i)) > 0).astype(np.float32)
      ax4.fill_between(frame_t, offset_y, offset_y + bit_active * 0.8,
                       step='post', alpha=0.6,
                       label=f'KS[{i}] {_KS_NAMES[i]}')
      offset_y += 1.0

    if not active_ks_indices:
      ax4.text(0.5, 0.5, 'Nessun KS attivo in questa finestra',
               ha='center', va='center', transform=ax4.transAxes,
               fontsize=10, color='gray')

    # Overlay: active note MIDI pitch (normalizzato per la visualizzazione).
    pitch_norm = (note_midi - BassMidiKeymap.NOTE_MIDI_MIN) / float(
        BassMidiKeymap.NOTE_MIDI_MAX - BassMidiKeymap.NOTE_MIDI_MIN)
    pitch_norm[note_midi == 0] = np.nan   # silenzio -> nan (non plottato)
    ax4_r = ax4.twinx()
    ax4_r.plot(frame_t, note_midi, color='navy', linewidth=1.2, alpha=0.7,
               label='active_note_midi (MIDI #)')
    ax4_r.set_ylabel('MIDI pitch (21-64)')
    ax4_r.set_ylim(20, 65)
    ax4_r.legend(loc='upper right', fontsize=8)

    ax4.set_xlim(t0, t_end)
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('KS index (stack)')
    ax4.set_title(
        f'Plot 4 — Finestra {iw}: active_ks_bits (stackato) '
        f'+ active_note_midi (blu). '
        f'KS attivi: {[_KS_NAMES[i] for i in active_ks_indices] or "nessuno"}')
    if active_ks_indices:
      ax4.legend(loc='upper left', fontsize=7)
    ax4.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig(args.output, dpi=110)
  print(f'\n[ok] saved figure -> {args.output}')


if __name__ == '__main__':
  main()
