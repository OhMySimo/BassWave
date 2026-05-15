# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Misura l'offset audio-MIDI sull'intero dataset BassWave.

Per ogni coppia (FLAC, MIDI) in metadata.jsonl:
  1. Parse MIDI con override_bpm=120 (renderer normalizza tutto a 120 BPM).
  2. Detect audio onset con librosa.onset.onset_detect.
  3. Per ogni MIDI onset, trova il nearest audio onset entro tolleranza
     (default 100 ms).
  4. Calcola lag = audio_onset - midi_onset.
  5. Aggrega per-file: median(lag), std(lag), n_matched/n_midi.

Output:
  * stdout: progress bar + statistiche aggregate (mean/std/percentili).
  * --out_jsonl: una riga per file con
      {midi_path, flac_path, n_midi, n_matched, median_lag_ms,
       std_lag_ms, slope_ms_per_s, intercept_ms}
    Utile per identificare file con allineamento anomalo (outlier).

Note di velocita':
  * librosa onset detection ~50-200 ms per file 16s (CPU).
  * 26k file = ~30-90 min totale.
  * Parallelizzabile via --n_workers (default 1, usa multiprocessing).

Esempio:
  python measure_midi_offset.py \\
      --metadata_jsonl /media/simone/NVME/MidiDataset/metadata.jsonl \\
      --src_prefix 'E:/' \\
      --dst_prefix '/media/simone/NVME/' \\
      --out_jsonl /tmp/offset_measurements.jsonl \\
      --n_workers 8 \\
      --max_files 0
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Path module-level so multiprocessing workers can import.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)


# -----------------------------------------------------------------------------
# Per-file measurement.
# -----------------------------------------------------------------------------


def _detect_audio_onsets(audio: np.ndarray, sr: int,
                          hop_ms: float = 10.0,
                          threshold: float = 0.12,
                          min_sep_ms: float = 40.0) -> np.ndarray:
  """Onset detector leggero: spectral flux su finestre piccole.

  Niente librosa, niente STFT grande. Usa rfft su blocchi da 512 sample
  (hop=441 sample @ 44.1k = 10ms). Peak-picking con soglia adattiva e
  separazione minima.

  Allocazione massima per file 16.5s: mags=[1650, 257] float32 = ~1.7MB.
  """
  hop = max(1, int(sr * hop_ms / 1000))
  fft_size = 512
  win = np.hanning(fft_size).astype(np.float32)
  n_frames = (len(audio) - fft_size) // hop
  if n_frames < 2:
    return np.array([], dtype=np.float64)

  # Compute log-magnitude spectral flux in one pass, frame by frame,
  # keeping only two frames in memory at a time to minimise peak RAM.
  prev_mag = np.abs(np.fft.rfft(audio[:fft_size] * win))
  flux = np.empty(n_frames - 1, dtype=np.float32)
  for i in range(1, n_frames):
    seg = audio[i * hop: i * hop + fft_size]
    if len(seg) < fft_size:
      seg = np.pad(seg, (0, fft_size - len(seg)))
    cur_mag = np.abs(np.fft.rfft(seg.astype(np.float32) * win))
    flux[i - 1] = np.maximum(cur_mag - prev_mag, 0).sum()
    prev_mag = cur_mag

  if flux.max() < 1e-9:
    return np.array([], dtype=np.float64)
  flux /= flux.max()

  # Simple adaptive threshold + minimum separation peak picking.
  min_sep = max(1, int(min_sep_ms / hop_ms))
  times = []
  last_pk = -min_sep
  for i in range(len(flux)):
    if flux[i] > threshold and i - last_pk >= min_sep:
      # Confirm it's a local max within ±2 frames.
      lo, hi = max(0, i - 2), min(len(flux), i + 3)
      if flux[i] >= flux[lo:hi].max() - 1e-6:
        times.append((i + 1) * hop_ms / 1000.0)  # +1 = flux is diff of frames
        last_pk = i
  return np.array(times, dtype=np.float64)


def _is_silent(flac_path: str, rms_threshold: float = 1e-4) -> bool:
  """Legge solo i primi 2s di audio e controlla RMS. Veloce (~5ms/file).

  Se RMS < threshold il file e' vuoto/silenzioso (render fallito).
  Threshold 1e-4 = -80 dBFS, ben sotto qualsiasi segnale reale di basso.
  """
  import soundfile as sf
  with sf.SoundFile(flac_path) as f:
    sr = f.samplerate
    # Leggi solo i primi 2s (max 88200 sample a 44.1k).
    frames_to_read = min(int(2.0 * sr), len(f))
    audio = f.read(frames=frames_to_read, dtype='float32', always_2d=False)
  if audio.ndim > 1:
    audio = audio.mean(axis=1)
  rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
  return rms < rms_threshold


def _midi_onsets_at_bpm(midi_path: str, target_bpm: float) -> np.ndarray:
  """Restituisci onset MIDI (s) assumendo tempo=target_bpm costante.

  Ignora i set_tempo event del MIDI. Sec/tick = 60 / (bpm * tpb).
  """
  import mido
  mf = mido.MidiFile(midi_path)
  tpb = mf.ticks_per_beat
  sec_per_tick = 60.0 / (target_bpm * tpb)
  onsets = []
  for tr in mf.tracks:
    cum = 0
    for msg in tr:
      cum += msg.time
      if msg.type == 'note_on' and msg.velocity > 0:
        onsets.append(cum * sec_per_tick)
  return np.array(sorted(set(onsets)), dtype=np.float64)


def measure_file(args: Tuple[str, str, float, float]) -> dict:
  """Worker: misura offset per una coppia (FLAC, MIDI).

  Args:
    args: tupla (midi_path, flac_path, target_bpm, match_tol_ms).

  Returns:
    dict con stat (vedi docstring file).
  """
  midi_path, flac_path, target_bpm, tol_ms = args
  out = {
      'midi_path': midi_path,
      'flac_path': flac_path,
      'n_midi': 0,
      'n_matched': 0,
      'median_lag_ms': None,
      'std_lag_ms': None,
      'mean_lag_ms': None,
      'slope_ms_per_s': None,
      'intercept_ms': None,
      'flac_dur_s': None,
      'error': None,
  }
  try:
    import soundfile as sf

    # Fast silence check BEFORE loading full audio.
    if _is_silent(flac_path):
      out['error'] = 'silent_file'
      return out

    audio, sr = sf.read(flac_path, dtype='float32', always_2d=False)
    if audio.ndim > 1:
      audio = audio.mean(axis=1)
    out['flac_dur_s'] = float(len(audio) / sr)

    audio_on = _detect_audio_onsets(audio, sr)
    midi_on = _midi_onsets_at_bpm(midi_path, target_bpm)
    out['n_midi'] = int(len(midi_on))
    if len(audio_on) == 0 or len(midi_on) == 0:
      out['error'] = 'no_onsets'
      return out

    tol_s = tol_ms / 1000.0
    matched_midi = []
    matched_lag = []
    for mt in midi_on:
      d = np.abs(audio_on - mt)
      i = int(np.argmin(d))
      if d[i] < tol_s:
        matched_midi.append(mt)
        matched_lag.append(audio_on[i] - mt)

    if not matched_lag:
      out['error'] = 'no_match'
      return out

    matched_midi = np.array(matched_midi)
    matched_lag_ms = np.array(matched_lag) * 1000.0

    out['n_matched'] = int(len(matched_lag_ms))
    out['median_lag_ms'] = float(np.median(matched_lag_ms))
    out['mean_lag_ms'] = float(np.mean(matched_lag_ms))
    out['std_lag_ms'] = float(np.std(matched_lag_ms))

    # Linear drift estimate: lag = slope*t + intercept (both in ms).
    if len(matched_lag_ms) >= 5:
      slope, intercept = np.polyfit(matched_midi, matched_lag_ms, 1)
      out['slope_ms_per_s'] = float(slope)
      out['intercept_ms'] = float(intercept)
    return out
  except Exception as e:
    out['error'] = f'{type(e).__name__}: {e}'
    return out


# -----------------------------------------------------------------------------
# Path translation (Windows -> Linux for BassWave).
# -----------------------------------------------------------------------------


def _translate(p: str, src: str, dst: str) -> str:
  p = p.replace('\\', '/')
  return dst + p[len(src):] if p.startswith(src) else p


# -----------------------------------------------------------------------------
# Driver.
# -----------------------------------------------------------------------------


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--metadata_jsonl', required=True)
  ap.add_argument('--src_prefix', default='E:/',
                  help='Path prefix da sostituire (Windows root).')
  ap.add_argument('--dst_prefix', default='/media/simone/NVME/',
                  help='Path prefix di destinazione (Linux root).')
  ap.add_argument('--out_jsonl', default='./midi_offset_measurements.jsonl')
  ap.add_argument('--target_bpm', type=float, default=120.0,
                  help='BPM target (renderer = 120 per BassWave).')
  ap.add_argument('--match_tol_ms', type=float, default=100.0,
                  help='Tolleranza max per accoppiare audio<->MIDI onset.')
  ap.add_argument('--n_workers', type=int, default=1,
                  help='Worker paralleli (default 1, =0 disabilita mp).')
  ap.add_argument('--max_files', type=int, default=0,
                  help='Limita a N file per test rapido (0=tutti).')
  ap.add_argument('--n_stratified', type=int, default=0,
                  help='Campiona N file in modo stratificato per (bpm, pack). '
                       '0 = disabilitato. Usa insieme a --max_files=0. '
                       'Esempio: 1000 da 26k = copertura ~4%% per strato.')
  ap.add_argument('--report_every', type=int, default=500,
                  help='Print progress ogni N file.')
  ap.add_argument('--checkpoint_every', type=int, default=200,
                  help='Salva risultati parziali ogni N file (default 200). '
                       'Se lo script crasha, al riavvio salta i file gia\'  '
                       'processati leggendo il file --out_jsonl esistente.')
  args = ap.parse_args()

  # ---- Read metadata, build worker args list ----
  print(f'[meta] reading {args.metadata_jsonl}...', flush=True)
  worker_args = []
  with open(args.metadata_jsonl) as f:
    for line in f:
      d = json.loads(line)
      midi = _translate(d['midi_path'], args.src_prefix, args.dst_prefix)
      flac = _translate(d['flac_path'], args.src_prefix, args.dst_prefix)
      worker_args.append(
          (midi, flac, args.target_bpm, args.match_tol_ms))
  n_total = len(worker_args)
  print(f'[meta] {n_total} (FLAC, MIDI) pairs found.')

  if args.n_stratified > 0 and args.max_files == 0:
    # Stratified sampling: read bpm+pack from metadata.jsonl for grouping.
    import random, math
    # Re-read metadata to get bpm and pack for each entry.
    meta_rows = []
    with open(args.metadata_jsonl) as _f:
      for _line in _f:
        meta_rows.append(json.loads(_line))
    # Build strata by (bpm_bucket, pack).
    from collections import defaultdict
    strata = defaultdict(list)
    for i, row in enumerate(meta_rows):
      bpm = int(row.get('bpm', 0) // 10 * 10)  # bucket to nearest 10 BPM
      pack = str(row.get('pack', 'unknown'))
      strata[(bpm, pack)].append(i)
    n_strata = len(strata)
    per_stratum = max(1, math.ceil(args.n_stratified / n_strata))
    selected = []
    for idxs in strata.values():
      k = min(per_stratum, len(idxs))
      selected.extend(random.sample(idxs, k))
    # Trim to exactly n_stratified.
    random.shuffle(selected)
    selected = selected[:args.n_stratified]
    worker_args = [worker_args[i] for i in selected]
    print(f'[meta] stratified sample: {len(worker_args)} pairs '
          f'from {n_strata} strata (bpm_bucket x pack), '
          f'~{per_stratum} per stratum.')

  if args.max_files > 0:
    worker_args = worker_args[:args.max_files]
    print(f'[meta] limited to {len(worker_args)} pairs (--max_files).')

  # ---- Resume: skip already-processed files ----
  already_done = set()
  results = []
  if os.path.exists(args.out_jsonl):
    with open(args.out_jsonl) as _f:
      for _line in _f:
        try:
          _r = json.loads(_line)
          already_done.add(_r['flac_path'])
          results.append(_r)
        except Exception:
          pass
    if already_done:
      n_before = len(worker_args)
      worker_args = [wa for wa in worker_args if wa[1] not in already_done]
      print(f'[resume] found {len(already_done)} already-processed files in '
            f'{args.out_jsonl}. Skipping, {len(worker_args)} remaining.')

  # ---- Run ----
  t_start = time.time()

  if args.n_workers <= 1:
    # Sequential — easier to debug.
    for i, wa in enumerate(worker_args):
      r = measure_file(wa)
      results.append(r)
      if (i + 1) % args.checkpoint_every == 0:
        # Checkpoint: append new results to out_jsonl.
        with open(args.out_jsonl, 'a') as _cf:
          for _r in results[len(already_done) + i + 1 - args.checkpoint_every:
                           len(already_done) + i + 1]:
            _cf.write(json.dumps(_r) + '\n')
      if (i + 1) % args.report_every == 0:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (len(worker_args) - i - 1) / rate
        n_silent = sum(1 for _r in results if _r.get('error') == 'silent_file')
        print(f'  [{i+1}/{len(worker_args)}] '
              f'{rate:.1f} files/s  ETA {eta/60:.1f} min  '
              f'silent_so_far={n_silent}',
              flush=True)
  else:
    # Parallel with multiprocessing pool.
    with mp.Pool(args.n_workers) as pool:
      for i, r in enumerate(pool.imap_unordered(
          measure_file, worker_args, chunksize=8)):
        results.append(r)
        if (i + 1) % args.report_every == 0:
          elapsed = time.time() - t_start
          rate = (i + 1) / elapsed
          eta = (len(worker_args) - i - 1) / rate
          print(f'  [{i+1}/{len(worker_args)}] '
                f'{rate:.1f} files/s  ETA {eta/60:.1f} min',
                flush=True)

  elapsed = time.time() - t_start
  print(f'\n[done] processed {len(results)} files in {elapsed/60:.1f} min '
        f'({len(results)/elapsed:.2f} files/s)')

  # ---- Write remaining (not yet checkpointed) results ----
  out_path = args.out_jsonl
  os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
  # Find how many new results were written in the last incomplete checkpoint.
  n_checkpointed = (len(results) - len(already_done)) // args.checkpoint_every * args.checkpoint_every
  remaining = results[len(already_done) + n_checkpointed:]
  with open(out_path, 'a') as f:
    for r in remaining:
      f.write(json.dumps(r) + '\n')
  print(f'[write] {out_path}  (total lines: {len(results)})')

  # ---- Silent file report ----
  silent = [r for r in results if r.get('error') == 'silent_file']
  if silent:
    silent_list_path = out_path.replace('.jsonl', '_silent_files.txt')
    with open(silent_list_path, 'w') as f:
      for r in silent:
        f.write(r['flac_path'] + '\n')
    print(f'[silent] {len(silent)} silent files -> {silent_list_path}')

  # ---- Aggregate statistics ----
  errors  = [r for r in results if r.get('error')]
  silent  = [r for r in results if r.get('error') == 'silent_file']
  no_on   = [r for r in results if r.get('error') == 'no_onsets']
  other_e = [r for r in results
             if r.get('error') and r.get('error') not in
             ('silent_file', 'no_onsets')]
  good = [r for r in results
          if r['error'] is None and r['median_lag_ms'] is not None]
  print(f'\n=== AGGREGATE STATS ===')
  print(f'  total           : {len(results)}')
  print(f'  good (matched)  : {len(good)} ({100*len(good)/len(results):.1f}%)')
  print(f'  silent files    : {len(silent)} ({100*len(silent)/len(results):.1f}%) <- skip these in training')
  print(f'  no_onsets       : {len(no_on)} ({100*len(no_on)/len(results):.1f}%) <- soft attack presets, usable')
  print(f'  other errors    : {len(other_e)} ({100*len(other_e)/len(results):.1f}%)')

  if other_e[:5]:
    print(f'  first 5 other errors:')
    for r in other_e[:5]:
      print(f'    {r["error"]}: {os.path.basename(r["flac_path"])}')

  if good:
    median_lags = np.array([r['median_lag_ms'] for r in good])
    std_lags    = np.array([r['std_lag_ms']    for r in good])
    slopes      = np.array([r['slope_ms_per_s']
                            for r in good
                            if r['slope_ms_per_s'] is not None])
    match_frac  = np.array([r['n_matched'] / max(r['n_midi'], 1)
                            for r in good])

    print(f'\nMedian-lag per file (ms) — distribution over {len(good)} files:')
    print(f'  mean      : {median_lags.mean():+8.2f}')
    print(f'  median    : {np.median(median_lags):+8.2f}')
    print(f'  std       : {median_lags.std():8.2f}')
    print(f'  p5/p95    : {np.percentile(median_lags, 5):+.2f} / '
          f'{np.percentile(median_lags, 95):+.2f}')
    print(f'  min/max   : {median_lags.min():+.2f} / {median_lags.max():+.2f}')

    print(f'\nStd-lag per file (ms; intra-file scatter) — over {len(good)} files:')
    print(f'  mean      : {std_lags.mean():.2f}')
    print(f'  median    : {np.median(std_lags):.2f}')
    print(f'  p95       : {np.percentile(std_lags, 95):.2f}')

    if len(slopes):
      print(f'\nDrift slope per file (ms/s) — over {len(slopes)} files:')
      print(f'  mean      : {slopes.mean():+.4f}')
      print(f'  median    : {np.median(slopes):+.4f}')
      print(f'  std       : {slopes.std():.4f}')
      print(f'  p5/p95    : {np.percentile(slopes, 5):+.4f} / '
            f'{np.percentile(slopes, 95):+.4f}')

    print(f'\nMatch fraction per file (n_matched/n_midi):')
    print(f'  mean      : {match_frac.mean():.3f}')
    print(f'  median    : {np.median(match_frac):.3f}')
    print(f'  p5        : {np.percentile(match_frac, 5):.3f}')

  print()


if __name__ == '__main__':
  main()
