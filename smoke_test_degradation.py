"""Smoke test for the BassWave degradation pipeline.

For each of N input FLAC files:
  1. Save original audio.
  2. Apply each degradation in isolation (all 6, wet=100) and save.
  3. Apply the full pipeline (all 6 active, wet=100) and save.
  4. Run CREPE on every version and compare f0 vs the clean baseline.

Metrics per (file, mode):
  median_abs_cents     — median |Δ cents| over voiced frames
  p90_abs_cents        — 90th percentile of |Δ cents|
  octave_error_rate    — fraction of voiced frames with |Δ cents| > 600
  rpa50, rpa100        — Raw Pitch Accuracy within ±50 / ±100 cents
  voiced_overlap_pct   — fraction of frames where both clean & degraded
                         have CREPE confidence > 0.5

Aggregate summary across files prints to stdout + writes f0_metrics.csv.

Usage:
  python smoke_test_degradation.py \\
    --input_root=/media/simone/NVME/MidiDataset/FLAC_AUG \\
    --output_dir=./smoke_test_output \\
    --n_files=5 \\
    --crepe_model_size=full \\
    --per_preset

Open output_dir/<file_id>/ in a file browser, listen to the WAVs, and read
the metrics CSV.
"""

import os

# Silencer prologue (must come before TF imports).
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('GLOG_minloglevel', '2')
import warnings
warnings.filterwarnings('ignore')

import argparse
import csv
import functools
import glob
import random
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import tensorflow.compat.v2 as tf

# Project imports — assumes script is run from the ddsp project root.
from ddsp.training.degradations.aggressive_compression import (
    AggressiveCompression)
from ddsp.training.degradations.bandwidth_limit import BandwidthLimit
from ddsp.training.degradations.degradation_main import DegradationPipeline
from ddsp.training.degradations.dynamic_eq import DynamicEQ
from ddsp.training.degradations.ghosting import Ghosting
from ddsp.training.degradations.phasing import Phasing
from ddsp.training.degradations.wrong_eq import WrongEQ


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--input_root', required=True)
  p.add_argument('--output_dir', default='./smoke_test_output')
  p.add_argument('--n_files', type=int, default=5)
  p.add_argument('--sample_rate', type=int, default=44100)
  p.add_argument('--max_secs', type=float, default=8.0,
                 help='Trim files to this length to keep CREPE calls fast.')
  p.add_argument('--crepe_model_size', default='full',
                 choices=['tiny', 'small', 'medium', 'large', 'full'])
  p.add_argument('--per_preset', action='store_true',
                 help='Stratify file sampling: one per preset where available.')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--no_plots', action='store_true',
                 help='Skip f0 contour plots (matplotlib not required).')
  return p.parse_args()


# -----------------------------------------------------------------------------
# IO + resampling.
# -----------------------------------------------------------------------------


def load_audio(path: str, target_sr: int) -> np.ndarray:
  audio, sr = sf.read(path, dtype='float32', always_2d=False)
  if audio.ndim > 1:
    audio = audio.mean(axis=1).astype(np.float32)
  if sr != target_sr:
    audio = resample(audio, sr, target_sr)
  return audio


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
  if src_sr == dst_sr:
    return audio.astype(np.float32, copy=False)
  try:
    import soxr
    return soxr.resample(audio, src_sr, dst_sr, quality='HQ').astype(
        np.float32)
  except ImportError:
    from fractions import Fraction
    from scipy.signal import resample_poly
    f = Fraction(dst_sr, src_sr).limit_denominator(10000)
    return resample_poly(
        audio, f.numerator, f.denominator).astype(np.float32)


def find_files(root: str, n: int, seed: int, stratify: bool) -> List[str]:
  flacs = glob.glob(os.path.join(root, '**', '*.flac'), recursive=True)
  if not flacs:
    raise RuntimeError(f'No FLAC files under {root}')
  rng = random.Random(seed)
  if not stratify:
    return rng.sample(flacs, min(n, len(flacs)))
  # Group by preset (suffix after "Pk" in filename).
  by_preset = {}
  for f in flacs:
    bn = os.path.basename(f)
    if '__Pk' in bn:
      preset = bn.split('__Pk')[1].rsplit('.flac', 1)[0]
      by_preset.setdefault(preset, []).append(f)
  chosen = [rng.choice(paths) for paths in by_preset.values()]
  rng.shuffle(chosen)
  return chosen[:n]


# -----------------------------------------------------------------------------
# Degradation modes.
# -----------------------------------------------------------------------------


def build_modes(sample_rate: int) -> Dict[str, callable]:
  """Return ordered dict {mode_name: fn(audio_44k_tensor) -> degraded_44k}.

  Each mode is a callable taking a [1, T] tensor and returning [1, T].
  Wet/dry forced to 100% so we can SEE the worst-case result. The pipeline
  mode forces all 6 layers to be applied (min/max_n_apply=6) and disables
  the clean passthrough.
  """
  modes = {
      'phasing_only': Phasing(
          wet_dry_pct=100.0, sample_rate=sample_rate),
      'ghosting_only': Ghosting(
          wet_dry_pct=100.0, sample_rate=sample_rate),
      'wrong_eq_only': WrongEQ(
          wet_dry_pct=100.0, sample_rate=sample_rate),
      'compression_only': AggressiveCompression(
          wet_dry_pct=100.0, sample_rate=sample_rate),
      'dynamic_eq_only': DynamicEQ(
          wet_dry_pct=100.0, sample_rate=sample_rate),
      'bandwidth_only': BandwidthLimit(
          wet_dry_pct=100.0, sample_rate=sample_rate),
      'full_pipeline': DegradationPipeline(
          sample_rate=sample_rate,
          phasing_wet=100, ghosting_wet=100, wrong_eq_wet=100,
          compression_wet=100, dynamic_eq_wet=100, bandwidth_wet=100,
          clean_passthrough_prob=0.0,
          min_n_apply=6, max_n_apply=6),
  }
  return {name: lambda x, layer=layer: layer(x, training=True)
          for name, layer in modes.items()}


def apply_mode(mode_fn, audio_np: np.ndarray) -> np.ndarray:
  """Apply a degradation mode to a numpy [T] audio array.

  Wraps in [1, T] tensor, calls the layer, returns numpy [T].
  """
  audio_t = tf.constant(audio_np[np.newaxis, :], dtype=tf.float32)
  out = mode_fn(audio_t)
  return out.numpy()[0]


# -----------------------------------------------------------------------------
# CREPE + f0 metrics.
# -----------------------------------------------------------------------------


_crepe_initialised = False
def init_crepe(model_size: str):
  global _crepe_initialised
  if _crepe_initialised:
    return
  import crepe
  orig = crepe.predict
  crepe.predict = functools.partial(orig, model_capacity=model_size)
  _crepe_initialised = True


def crepe_f0(audio_44k: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
  """Run CREPE on audio (resampled to 16k internally). Returns (f0, conf)."""
  import crepe
  audio_16k = resample(audio_44k, sr, 16000)
  # Tiny dither (matches prep) to avoid CREPE std-norm divide-by-zero on
  # any silent frame.
  audio_16k = audio_16k + np.random.RandomState(0).randn(
      len(audio_16k)).astype(np.float32) * 1e-6
  _, f0, conf, _ = crepe.predict(
      audio_16k, sr=16000, viterbi=True, step_size=20, verbose=0)
  f0 = np.nan_to_num(f0, nan=0.0)
  conf = np.nan_to_num(conf, nan=0.0)
  return f0, conf


def f0_metrics(f0_clean, conf_clean, f0_deg, conf_deg,
               conf_threshold: float = 0.5) -> Dict[str, float]:
  """Compare two f0 sequences. Both expected at 50 Hz frame rate, same length.

  Returns dict of metrics. Missing metrics (e.g. no voiced frames) are NaN.
  """
  # Trim to the same length in case CREPE returned slightly different sizes.
  L = min(len(f0_clean), len(f0_deg))
  f0c = f0_clean[:L]
  f0d = f0_deg[:L]
  cc = conf_clean[:L]
  cd = conf_deg[:L]

  voiced_clean = (cc > conf_threshold) & (f0c > 20.0)
  voiced_deg = (cd > conf_threshold) & (f0d > 20.0)
  voiced_both = voiced_clean & voiced_deg

  voiced_overlap_pct = (
      voiced_both.mean() / max(voiced_clean.mean(), 1e-6) * 100.0)

  if voiced_both.sum() < 5:
    return {
        'voiced_clean_pct': 100.0 * voiced_clean.mean(),
        'voiced_deg_pct': 100.0 * voiced_deg.mean(),
        'voiced_overlap_pct': voiced_overlap_pct,
        'median_abs_cents': float('nan'),
        'p90_abs_cents': float('nan'),
        'octave_error_rate': float('nan'),
        'rpa50': float('nan'),
        'rpa100': float('nan'),
        'mean_clean_f0_hz': float(np.nan if not voiced_clean.any()
                                   else f0c[voiced_clean].mean()),
        'mean_deg_f0_hz': float(np.nan if not voiced_deg.any()
                                 else f0d[voiced_deg].mean()),
    }

  cents = 1200.0 * np.log2(
      f0d[voiced_both] / np.maximum(f0c[voiced_both], 1e-3))
  abs_cents = np.abs(cents)
  return {
      'voiced_clean_pct': 100.0 * voiced_clean.mean(),
      'voiced_deg_pct': 100.0 * voiced_deg.mean(),
      'voiced_overlap_pct': voiced_overlap_pct,
      'median_abs_cents': float(np.median(abs_cents)),
      'p90_abs_cents': float(np.percentile(abs_cents, 90)),
      'octave_error_rate': float((abs_cents > 600).mean()),
      'rpa50': float((abs_cents <= 50).mean()),
      'rpa100': float((abs_cents <= 100).mean()),
      'mean_clean_f0_hz': float(f0c[voiced_clean].mean()),
      'mean_deg_f0_hz': float(f0d[voiced_deg].mean()),
  }


# -----------------------------------------------------------------------------
# Plotting (optional).
# -----------------------------------------------------------------------------


def plot_f0_overlay(f0_clean, conf_clean, all_modes_f0,
                    out_path: str, title: str):
  """Overlay clean f0 and all-modes f0 contours; save PNG."""
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    return

  n_modes = len(all_modes_f0)
  fig, axes = plt.subplots(n_modes, 1, figsize=(12, 2 * n_modes), sharex=True)
  if n_modes == 1:
    axes = [axes]
  t = np.arange(len(f0_clean)) / 50.0   # 50 Hz frame rate
  voiced = (conf_clean > 0.5) & (f0_clean > 20)

  for ax, (mode, (f0_d, conf_d)) in zip(axes, all_modes_f0.items()):
    ax.plot(t[voiced], f0_clean[voiced], '.', color='steelblue',
            label='clean', markersize=2)
    L = min(len(t), len(f0_d))
    voiced_d = (conf_d[:L] > 0.5) & (f0_d[:L] > 20)
    ax.plot(t[:L][voiced_d], f0_d[:L][voiced_d], '.', color='crimson',
            label=mode, markersize=2, alpha=0.6)
    ax.set_ylabel('f0 [Hz]')
    ax.set_yscale('log')
    ax.set_ylim(20, 2000)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
  axes[-1].set_xlabel('time [s]')
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(out_path, dpi=100)
  plt.close(fig)


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------


def main():
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)

  init_crepe(args.crepe_model_size)
  print(f'CREPE model_capacity = {args.crepe_model_size}')

  modes = build_modes(args.sample_rate)
  print(f'Modes: {list(modes.keys())}')

  files = find_files(args.input_root, args.n_files, args.seed, args.per_preset)
  print(f'Selected {len(files)} files'
        f' ({"per-preset" if args.per_preset else "uniform"}).')
  print()

  # Per-row (file, mode) metrics → CSV.
  rows: List[Dict] = []

  for file_idx, path in enumerate(files):
    fname = os.path.basename(path).replace('.flac', '')
    file_id = f'{file_idx:02d}_{fname}'
    file_dir = os.path.join(args.output_dir, file_id)
    os.makedirs(file_dir, exist_ok=True)
    print(f'[{file_idx + 1}/{len(files)}] {fname}')

    # Load and trim.
    audio = load_audio(path, args.sample_rate)
    max_n = int(args.max_secs * args.sample_rate)
    if len(audio) > max_n:
      audio = audio[:max_n]

    # Save clean.
    sf.write(os.path.join(file_dir, '00_clean.wav'),
             audio, args.sample_rate, subtype='FLOAT')
    f0_clean, conf_clean = crepe_f0(audio, args.sample_rate)
    voiced_clean_pct = 100.0 * ((conf_clean > 0.5) & (f0_clean > 20)).mean()
    median_clean_f0 = float(np.median(
        f0_clean[(conf_clean > 0.5) & (f0_clean > 20)])) \
        if voiced_clean_pct > 0 else float('nan')
    print(f'    clean: voiced={voiced_clean_pct:5.1f}%  '
          f'median f0={median_clean_f0:6.1f} Hz')

    # Per-mode loop.
    f0_overlay = {}
    for mode_idx, (mode_name, mode_fn) in enumerate(modes.items()):
      try:
        deg = apply_mode(mode_fn, audio)
      except Exception as e:
        print(f'    [{mode_name}] degradation FAILED: {e}')
        continue

      out_wav = os.path.join(
          file_dir, f'{mode_idx + 1:02d}_{mode_name}.wav')
      sf.write(out_wav, np.clip(deg, -1.0, 1.0),
               args.sample_rate, subtype='FLOAT')

      f0_d, conf_d = crepe_f0(deg, args.sample_rate)
      m = f0_metrics(f0_clean, conf_clean, f0_d, conf_d)
      f0_overlay[mode_name] = (f0_d, conf_d)

      m_pretty = (
          f'voiced_overlap={m["voiced_overlap_pct"]:5.1f}%  '
          f'med|Δcents|={m["median_abs_cents"]:6.1f}  '
          f'p90|Δcents|={m["p90_abs_cents"]:6.1f}  '
          f'octave_err={m["octave_error_rate"]*100:5.1f}%  '
          f'RPA50={m["rpa50"]*100:5.1f}%')
      print(f'    [{mode_name:18s}] {m_pretty}')

      rows.append({
          'file_id': file_id,
          'mode': mode_name,
          'preset': fname.split('__Pk')[-1] if '__Pk' in fname else 'unknown',
          **m,
      })

    if not args.no_plots and f0_overlay:
      plot_f0_overlay(
          f0_clean, conf_clean, f0_overlay,
          out_path=os.path.join(file_dir, 'f0_overlay.png'),
          title=f'f0 contour: clean (blue) vs degraded (red) — {fname}')

    print()

  # ---------------------------------------------------------------------------
  # Aggregate metrics across files, per mode.
  # ---------------------------------------------------------------------------

  print('=' * 90)
  print('AGGREGATE (median across files, per mode):')
  print('=' * 90)
  hdr = (f'{"mode":20s}  {"voiced_overlap":>14s}  {"med|Δcents|":>11s}  '
         f'{"p90|Δcents|":>11s}  {"octave_err":>10s}  {"RPA50":>6s}  '
         f'{"RPA100":>7s}')
  print(hdr)
  print('-' * 90)
  by_mode: Dict[str, List[Dict]] = {}
  for r in rows:
    by_mode.setdefault(r['mode'], []).append(r)
  for mode, mrows in by_mode.items():
    def med(k):
      vals = [r[k] for r in mrows if not np.isnan(r[k])]
      return float(np.median(vals)) if vals else float('nan')
    print(f'{mode:20s}  '
          f'{med("voiced_overlap_pct"):13.1f}%  '
          f'{med("median_abs_cents"):11.1f}  '
          f'{med("p90_abs_cents"):11.1f}  '
          f'{med("octave_error_rate")*100:9.1f}%  '
          f'{med("rpa50")*100:5.1f}%  '
          f'{med("rpa100")*100:6.1f}%')

  # CSV with all rows.
  csv_path = os.path.join(args.output_dir, 'f0_metrics.csv')
  if rows:
    with open(csv_path, 'w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)
  print()
  print(f'WAV files + plots: {args.output_dir}/<file_id>/')
  print(f'Detailed CSV:      {csv_path}')

  # ---------------------------------------------------------------------------
  # Verdict heuristic.
  # ---------------------------------------------------------------------------

  print()
  print('=' * 90)
  print('VERDICT (per-mode, judged on octave_error_rate):')
  print('=' * 90)
  for mode, mrows in by_mode.items():
    oer = [r['octave_error_rate'] for r in mrows
           if not np.isnan(r['octave_error_rate'])]
    if not oer:
      print(f'  {mode:20s}  insufficient voiced data — INCONCLUSIVE')
      continue
    median_oer = float(np.median(oer)) * 100.0
    if median_oer < 2.0:
      verdict = '✓ SAFE for training'
    elif median_oer < 10.0:
      verdict = '⚠ marginal — consider reducing wet/dry for this layer'
    else:
      verdict = '✗ DANGEROUS — CREPE breaks; reduce wet or exclude from pipeline'
    print(f'  {mode:20s}  median octave-err={median_oer:5.1f}%  →  {verdict}')


if __name__ == '__main__':
  main()
