"""Sanity-check CREPE f0 quality on a sample of BassWave FLACs.

Bypasses TFRecord entirely — runs CREPE directly on raw FLAC files. Use
this to verify f0 quality with `--crepe_model_size=medium` BEFORE
committing to a 4-hour batch prep, or independently while a prep is
still running and TFRecord shards are partially written.

Usage:
  python check_crepe_f0.py \
    --input_root=/media/simone/NVME/MidiDataset/FLAC_AUG \
    --n_files=20 \
    --crepe_model_size=medium
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import warnings
warnings.filterwarnings('ignore')

import argparse
import functools
import glob
import random

import numpy as np
import soundfile as sf
import crepe


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_root', required=True,
                   help='Root of the BassWave FLAC dataset.')
    p.add_argument('--n_files', type=int, default=20,
                   help='Number of files to sample.')
    p.add_argument('--crepe_model_size', default='medium',
                   choices=['tiny', 'small', 'medium', 'large', 'full'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--per_preset', action='store_true',
                   help='Stratify sampling by preset (one file per preset '
                        'where possible) instead of uniform random.')
    return p.parse_args()


def find_flacs(root, n, seed, stratify_by_preset):
    flacs = glob.glob(os.path.join(root, '**', '*.flac'), recursive=True)
    if not flacs:
        raise RuntimeError(f'No FLACs under {root}')
    rng = random.Random(seed)
    if not stratify_by_preset:
        return rng.sample(flacs, min(n, len(flacs)))
    # Stratify: one per preset (preset is the suffix after "Pk" in filename).
    by_preset = {}
    for f in flacs:
        bn = os.path.basename(f)
        if '__Pk' in bn:
            preset = bn.split('__Pk')[1].rsplit('.flac', 1)[0]
            by_preset.setdefault(preset, []).append(f)
    chosen = []
    for preset, paths in by_preset.items():
        chosen.append(rng.choice(paths))
    rng.shuffle(chosen)
    return chosen[:n]


def main():
    args = parse_args()

    # Force the CREPE model size we want to test.
    _orig = crepe.predict
    crepe.predict = functools.partial(_orig, model_capacity=args.crepe_model_size)
    print(f'Using CREPE model_capacity={args.crepe_model_size}')

    files = find_flacs(args.input_root, args.n_files, args.seed,
                       args.per_preset)
    print(f'Sampled {len(files)} files'
          f' ({"stratified per preset" if args.per_preset else "uniform"}).')
    print()

    results = []
    for path in files:
        audio, sr = sf.read(path, dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        # Resample to 16k for CREPE.
        if sr != 16000:
            try:
                import soxr
                audio = soxr.resample(audio, sr, 16000, quality='HQ').astype(np.float32)
            except ImportError:
                from scipy.signal import resample_poly
                from fractions import Fraction
                f = Fraction(16000, sr).limit_denominator(10000)
                audio = resample_poly(audio, f.numerator, f.denominator).astype(np.float32)

        # Tiny dither to avoid CREPE's frame-stdev div-by-zero.
        audio = audio + np.random.randn(len(audio)).astype(np.float32) * 1e-6

        # Run CREPE.
        _, f0, conf, _ = crepe.predict(
            audio, sr=16000, viterbi=True, step_size=20, verbose=0)
        f0 = np.nan_to_num(f0, nan=0.0)
        conf = np.nan_to_num(conf, nan=0.0)
        voiced = (conf > 0.5) & (f0 > 20)
        f0_v = f0[voiced]

        if len(f0_v) < 10:
            print(f'  {os.path.basename(path):70s}  '
                  f'unvoiced (conf too low — silent/noisy file)')
            continue
        med = float(np.median(f0_v))
        p10, p90 = np.percentile(f0_v, [10, 90])
        results.append(med)
        # Octave-error heuristic: if median is suspiciously >100 Hz it
        # *might* be locked to the 2nd harmonic of a sub-bass note.
        flag = '  ⚠ check octave' if med > 130.0 else ''
        print(f'  {os.path.basename(path):70s}  '
              f'med={med:6.1f} Hz  [10..90]={p10:5.1f}..{p90:5.1f}  '
              f'voiced={voiced.mean()*100:.0f}%{flag}')

    if not results:
        print('\nNo files yielded a voiced segment — something is off.')
        return

    print()
    print(f'== Summary across {len(results)} files ==')
    print(f'  Overall F0 median:        {np.median(results):.1f} Hz')
    print(f'  F0 medians 10..90 pct:    '
          f'{np.percentile(results, 10):.1f} .. '
          f'{np.percentile(results, 90):.1f} Hz')
    print()
    if 50 <= np.median(results) <= 130:
        print('  ✓ Plausible bass range. CREPE-{} looks fine.'.format(
            args.crepe_model_size))
    elif np.median(results) > 130:
        print('  ⚠ Median is high. CREPE may be locking onto 2nd harmonic '
              'instead of the fundamental. Try --crepe_model_size=full, '
              'or switch to pyin/yin for f0 estimation.')
    else:
        print('  ⚠ Median is below typical bass range (expected ~50-130 Hz).')


if __name__ == '__main__':
    main()
