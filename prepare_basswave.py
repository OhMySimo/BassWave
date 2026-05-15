# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
r"""Prepare TFRecord shards from the BassWave FLAC dataset.

Walks the dataset directory tree:

  ROOT/<groove_cat>/<groove_pat>/<bpm-Sxxx@beats>/Variation_NN__Tp+N__Pk<preset>.flac

For each FLAC: load at 44.1 k mono, resample to 16 k for CREPE, run f0 +
loudness, slice into windows, write TFRecord examples that include audio +
features + metadata (preset_id, transpose, groove_cat_id).

Usage:
  python -m ddsp.training.data_preparation.prepare_basswave \
    --input_root=/media/simone/NVME/MidiDataset/FLAC_AUG \
    --output_dir=/media/simone/NVME/MidiDataset/BassWave_TFR \
    --num_shards=64 \
    --eval_split_fraction=0.05

Notes for low-F0 bass (BassWave median ~65 Hz):
  - viterbi=True is critical (smooths octave jumps).
  - frame_rate=50 Hz at 44.1 k = hop ≈ 882 samples (20 ms). Matches vst_48k.
  - We keep audio_16k (for CREPE rerun if ever needed at inference).
"""

# -----------------------------------------------------------------------------
# Log + warning silencer. MUST run before any heavy import (tensorflow, librosa,
# apache_beam, ...) — TF in particular emits its C++ INFO lines at
# import time, before Python's `warnings` module can do anything.
#
# Override by setting TF_CPP_MIN_LOG_LEVEL=0 (or unsetting any of the env
# vars below) in the shell.
# -----------------------------------------------------------------------------
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')   # 0=all, 1=I, 2=I+W, 3=all
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # quieter oneDNN init
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '2')

import warnings
# google.api_core FutureWarning — Python 3.10 EOL notice (irrelevant here).
warnings.filterwarnings('ignore', category=FutureWarning,
                        module=r'google\.api_core.*')
# apache_beam UserWarning — pkg_resources deprecation (out of our control).
warnings.filterwarnings('ignore', category=UserWarning,
                        module=r'apache_beam.*')
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module=r'apache_beam.*')
# librosa A-weighting log10(0) at DC bin — mathematically correct (curve
# is -inf dB at f=0), purely cosmetic warning.
warnings.filterwarnings('ignore',
                        message='divide by zero encountered in log10')
# CREPE per-frame normalisation — should not fire after the dither we add
# in compute_features(), but suppress defensively in case a degenerate
# file slips through.
warnings.filterwarnings('ignore',
                        message='invalid value encountered in divide',
                        module=r'crepe.*')

# Lower the absl/TF Python loggers (independent of the C++ flag above).
import logging as _py_logging
_py_logging.getLogger('tensorflow').setLevel(_py_logging.ERROR)
_py_logging.getLogger('absl').setLevel(_py_logging.INFO)  # keep absl INFO,
                                                          # we use it for our
                                                          # own progress logs.

# -----------------------------------------------------------------------------
# Now the real imports.
# -----------------------------------------------------------------------------
import hashlib
import re
from typing import Dict, Iterable, List, Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

# Direct imports — running this script requires the project on PYTHONPATH.
from ddsp import spectral_ops


FLAGS = flags.FLAGS

flags.DEFINE_string('input_root', None,
                    'Root of the BassWave dataset (FLAC_AUG dir).')
flags.DEFINE_string('output_dir', None,
                    'Where to write TFRecord shards.')
flags.DEFINE_integer('num_shards', 64, 'Number of TFRecord shards.')
flags.DEFINE_integer('sample_rate', 44100, 'Target audio SR.')
flags.DEFINE_integer('frame_rate', 50, 'Frame rate for f0/ld features.')
flags.DEFINE_float('example_secs', 4.0, 'Window length in seconds.')
flags.DEFINE_float('hop_secs', 2.0, 'Hop between consecutive windows.')
flags.DEFINE_float('eval_split_fraction', 0.05,
                   'Fraction of files held out for eval.')
flags.DEFINE_bool('centered', True,
                  'Centered framing for f0 (must match data provider).')
flags.DEFINE_bool('viterbi', True,
                  'Use Viterbi smoothing in CREPE (recommended for bass).')
flags.DEFINE_float('min_audio_rms_db', -50.0,
                   'Skip files whose RMS is below this dBFS threshold. '
                   'CREPE on near-silent audio produces NaN frames that can '
                   'crash TF on some platforms.')
flags.DEFINE_integer('log_every', 50,
                     'Log progress every N files (also makes the file that '
                     'triggers a segfault visible in the log).')


# -----------------------------------------------------------------------------
# Filename / path parsing.
# -----------------------------------------------------------------------------

# Variation_03__Tp+2__Pkmulti_fx2.flac  →  variation=3, transpose=+2, preset=multi_fx2
FILENAME_RE = re.compile(
    r'^Variation_(\d+)__Tp([+-]?\d+)__Pk(.+)\.flac$')


def parse_path(path: str, root: str) -> Optional[Dict]:
  """Parse a FLAC path into metadata. Returns None on layout mismatch.

  Note: preset_id and groove_cat_id are NOT assigned here — they're filled in
  later by `assign_ids` after the full preset/groove inventory is known.
  """
  rel = os.path.relpath(path, root)
  parts = rel.split(os.sep)
  if len(parts) < 4:
    return None
  groove_cat, groove_pat, bpm_dir, fname = parts[-4:]
  m = FILENAME_RE.match(fname)
  if not m:
    return None
  return {
      'path': path,
      'preset': m.group(3),
      'transpose': int(m.group(2)),
      'variation': int(m.group(1)),
      'groove_cat': groove_cat,
      'groove_pat': groove_pat,
      'bpm_dir': bpm_dir,
  }


def walk_dataset(root: str) -> List[Dict]:
  """Find all valid FLAC files under root."""
  records = []
  for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
      if not fname.endswith('.flac'):
        continue
      meta = parse_path(os.path.join(dirpath, fname), root)
      if meta is not None:
        records.append(meta)
  return records


def build_manifest(records: List[Dict]) -> Dict:
  """Discover the full preset / groove_cat inventory and assign stable IDs.

  IDs are assigned in alphabetical order so they're deterministic across runs
  even if you re-discover after adding new files. Counts are included for
  the data provider's preset-balanced sampling.
  """
  presets, grooves = set(), set()
  preset_counts, groove_counts = {}, {}
  for r in records:
    presets.add(r['preset'])
    grooves.add(r['groove_cat'])
    preset_counts[r['preset']] = preset_counts.get(r['preset'], 0) + 1
    groove_counts[r['groove_cat']] = groove_counts.get(r['groove_cat'], 0) + 1
  preset_list = sorted(presets)
  groove_list = sorted(grooves)
  return {
      'presets': preset_list,
      'preset_counts': preset_counts,
      'groove_cats': groove_list,
      'groove_cat_counts': groove_counts,
      'preset_to_id': {p: i for i, p in enumerate(preset_list)},
      'groove_cat_to_id': {g: i for i, g in enumerate(groove_list)},
  }


def assign_ids(records: List[Dict], manifest: Dict) -> List[Dict]:
  """In-place fill of preset_id / groove_cat_id based on the manifest."""
  p2i = manifest['preset_to_id']
  g2i = manifest['groove_cat_to_id']
  for r in records:
    r['preset_id'] = p2i[r['preset']]
    r['groove_cat_id'] = g2i[r['groove_cat']]
  return records


# -----------------------------------------------------------------------------
# Per-file processing.
# -----------------------------------------------------------------------------


def load_flac_once(path: str):
  """Load a FLAC at its native sample rate as float32 mono.

  Uses libsndfile via the `soundfile` package. NO subprocess, NO ffmpeg —
  this fixes two problems in the previous pydub-based loader:
    1. pydub spawned an ffmpeg subprocess per call (we previously called
       load_flac twice per file, so 2 subprocesses × 25k files).
    2. After Ctrl+C, ffmpeg subprocesses could be orphaned, requiring
       manual cleanup.

  Returns (audio, native_sr) at the file's native sample rate. Use
  resample_to() to convert to the desired target rate.
  """
  import soundfile as sf
  audio, native_sr = sf.read(path, dtype='float32', always_2d=False)
  if audio.ndim > 1:
    # Downmix to mono if the file happens to be stereo.
    audio = audio.mean(axis=1).astype(np.float32)
  return audio, int(native_sr)


def resample_to(audio: np.ndarray, native_sr: int,
                target_sr: int) -> np.ndarray:
  """High-quality resampling. Uses soxr if installed (5-10× faster than
  scipy on irrational ratios like 44100→16000), else falls back to
  scipy.signal.resample_poly.

  No-op if native_sr == target_sr.
  """
  if native_sr == target_sr:
    return audio.astype(np.float32, copy=False)
  try:
    import soxr
    return soxr.resample(audio, native_sr, target_sr,
                         quality='HQ').astype(np.float32)
  except ImportError:
    from fractions import Fraction
    from scipy.signal import resample_poly
    f = Fraction(target_sr, native_sr).limit_denominator(10000)
    return resample_poly(
        audio, f.numerator, f.denominator).astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
  """Full-track RMS in dBFS."""
  rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-20))
  return 20.0 * np.log10(rms + 1e-20)


def dither(audio: np.ndarray, level: float = 1e-6,
           seed: Optional[int] = None) -> np.ndarray:
  """Add inaudible Gaussian dither.

  Why we need this: CREPE normalises each input frame by `frames /=
  np.std(frames, axis=1)`. If even ONE frame in the audio has zero variance
  (digital silence, or a sample held constant for >25 ms — perfectly possible
  in a sustained bass note's release tail), std=0 → divide produces NaN/Inf.
  Those NaNs propagate into the TF graph and have been observed to
  segfault the CREPE inference on some TF/MKL/ROCm combinations.

  1e-6 amplitude is well below 24-bit noise floor (~-144 dBFS) and below
  16-bit floor (~-96 dBFS), so this is fully inaudible.
  """
  rng = np.random.RandomState(seed)
  return audio + rng.randn(len(audio)).astype(audio.dtype) * level


def scrub_nan(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
  """Replace NaN/±Inf with fill_value."""
  return np.nan_to_num(
      arr, nan=fill_value, posinf=fill_value, neginf=fill_value)


def compute_features(audio_44k: np.ndarray, audio_16k: np.ndarray,
                     frame_rate: int, viterbi: bool, centered: bool):
  """Compute f0 (CREPE on 16k) and loudness (on 16k for stability).

  Note: ddsp.spectral_ops.compute_f0 hard-codes CREPE model_capacity='full'
  via crepe.predict()'s default — there's no public knob to switch it. If
  you need a different size, monkey-patch crepe.predict before calling this.
  """
  padding = 'center' if centered else 'same'

  # Apply tiny dither to 16k stream BEFORE CREPE to prevent zero-variance
  # frames blowing up CREPE's per-frame normalisation. Seeded by the audio
  # content so re-runs are deterministic.
  seed = int(np.uint32(audio_16k[:64].tobytes().__hash__() & 0xFFFFFFFF))
  audio_16k_dithered = dither(audio_16k, level=1e-6, seed=seed)

  f0_hz, f0_conf = spectral_ops.compute_f0(
      audio_16k_dithered, frame_rate, viterbi=viterbi, padding=padding)
  loudness_db = spectral_ops.compute_loudness(
      audio_16k_dithered, sample_rate=spectral_ops.CREPE_SAMPLE_RATE,
      frame_rate=frame_rate, padding=padding)

  # Scrub any remaining NaN/Inf — defensive, in case CREPE produces them
  # for other reasons (extremely low confidence frames, etc).
  f0_hz = scrub_nan(np.asarray(f0_hz), fill_value=0.0).astype(np.float32)
  f0_conf = scrub_nan(np.asarray(f0_conf), fill_value=0.0).astype(np.float32)
  loudness_db = scrub_nan(
      loudness_db.numpy(), fill_value=-120.0).astype(np.float32)

  return f0_hz, f0_conf, loudness_db


def slice_into_windows(audio: np.ndarray, window_samples: int,
                       hop_samples: int) -> List[np.ndarray]:
  """Slice 1D audio into overlapping windows; drop short tail."""
  out = []
  i = 0
  while i + window_samples <= len(audio):
    out.append(audio[i:i + window_samples])
    i += hop_samples
  return out


def make_example(audio_44k_win, audio_16k_win, f0_win, conf_win, ld_win,
                 meta) -> tf.train.Example:
  """Construct a tf.train.Example from a single window + metadata."""
  fhash = int(hashlib.md5(meta['path'].encode()).hexdigest()[:15], 16)
  features = {
      'audio': tf.train.Feature(
          float_list=tf.train.FloatList(value=audio_44k_win)),
      'audio_16k': tf.train.Feature(
          float_list=tf.train.FloatList(value=audio_16k_win)),
      'f0_hz': tf.train.Feature(
          float_list=tf.train.FloatList(value=f0_win)),
      'f0_confidence': tf.train.Feature(
          float_list=tf.train.FloatList(value=conf_win)),
      'loudness_db': tf.train.Feature(
          float_list=tf.train.FloatList(value=ld_win)),
      'preset_id': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[meta['preset_id']])),
      'transpose': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[meta['transpose']])),
      'groove_cat_id': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[meta['groove_cat_id']])),
      'file_hash': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[fhash])),
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


# -----------------------------------------------------------------------------
# Main pipeline (single-process — for an Apache Beam version, see the notes
# at the bottom of this file).
# -----------------------------------------------------------------------------


_CREPE_FIRST_CALL = [True]   # mutable holder; set to False after first run.


def process_file(meta, sample_rate, frame_rate, example_secs, hop_secs,
                 viterbi, centered, hop_samples_native,
                 min_audio_rms_db=-50.0):
  """Process one FLAC into (list of tf.train.Example, timings dict).

  Loads the FLAC ONCE at native SR, then resamples to 44.1k and 16k. This
  halves the disk I/O and (more importantly) eliminates the per-file
  ffmpeg subprocess pair that pydub used to spawn — which was causing
  orphaned processes on Ctrl+C and slowdowns on slow disks.

  Skips files whose overall RMS is below `min_audio_rms_db` (default -50 dBFS,
  i.e. essentially silent). CREPE on silence produces NaN frames.
  """
  import time
  timings = {}
  if _CREPE_FIRST_CALL[0]:
    logging.info(
        '  Note: the first call to CREPE will load the model and may take '
        '1-3 minutes (including ROCm/CUDA JIT compilation on GPU). '
        'CPU usage will appear low during this time — this is normal. '
        'Subsequent files will be much faster.')

  t = time.perf_counter()
  audio_native, native_sr = load_flac_once(meta['path'])
  timings['load'] = time.perf_counter() - t

  t = time.perf_counter()
  audio_44k = resample_to(audio_native, native_sr, sample_rate)
  audio_16k = resample_to(
      audio_native, native_sr, spectral_ops.CREPE_SAMPLE_RATE)
  timings['resample'] = time.perf_counter() - t

  # Skip near-silent files: CREPE on silence is a recipe for NaN/segfaults.
  level_db = rms_db(audio_16k)
  if level_db < min_audio_rms_db:
    raise RuntimeError(
        f'audio RMS={level_db:.1f} dBFS < threshold {min_audio_rms_db} dBFS')

  t = time.perf_counter()
  f0, conf, ld = compute_features(
      audio_44k, audio_16k, frame_rate, viterbi, centered)
  timings['features'] = time.perf_counter() - t

  if _CREPE_FIRST_CALL[0]:
    logging.info('  CREPE initialized successfully — proceeding with batch.')
    _CREPE_FIRST_CALL[0] = False

  win_44k = int(example_secs * sample_rate)
  win_16k = int(example_secs * spectral_ops.CREPE_SAMPLE_RATE)
  win_feat = int(example_secs * frame_rate)
  if centered:
    win_44k += hop_samples_native
    win_16k += spectral_ops.CREPE_SAMPLE_RATE // frame_rate
    win_feat += 1

  hop_44k = int(hop_secs * sample_rate)
  hop_16k = int(hop_secs * spectral_ops.CREPE_SAMPLE_RATE)
  hop_feat = int(hop_secs * frame_rate)

  t = time.perf_counter()
  audio_wins_44k = slice_into_windows(audio_44k, win_44k, hop_44k)
  audio_wins_16k = slice_into_windows(audio_16k, win_16k, hop_16k)
  f0_wins = slice_into_windows(f0, win_feat, hop_feat)
  conf_wins = slice_into_windows(conf, win_feat, hop_feat)
  ld_wins = slice_into_windows(ld, win_feat, hop_feat)

  n = min(len(audio_wins_44k), len(audio_wins_16k), len(f0_wins),
          len(conf_wins), len(ld_wins))
  examples = []
  for i in range(n):
    ex = make_example(
        audio_wins_44k[i], audio_wins_16k[i],
        f0_wins[i], conf_wins[i], ld_wins[i],
        meta)
    examples.append(ex)
  timings['windowing'] = time.perf_counter() - t
  timings['n_windows'] = n
  return examples, timings


def main(_):
  if FLAGS.input_root is None or FLAGS.output_dir is None:
    raise ValueError('Must pass --input_root and --output_dir.')

  os.makedirs(FLAGS.output_dir, exist_ok=True)
  records = walk_dataset(FLAGS.input_root)
  logging.info('Found %d FLAC files.', len(records))
  if not records:
    raise RuntimeError('No FLAC files found.')

  # Discover preset / groove inventory and assign deterministic IDs.
  manifest = build_manifest(records)
  records = assign_ids(records, manifest)
  logging.info('Discovered %d presets, %d groove categories.',
               len(manifest['presets']), len(manifest['groove_cats']))
  for p in manifest['presets']:
    logging.info('  preset[%2d] %-32s  count=%d',
                 manifest['preset_to_id'][p], p,
                 manifest['preset_counts'][p])
  for g in manifest['groove_cats']:
    logging.info('  groove[%2d] %-32s  count=%d',
                 manifest['groove_cat_to_id'][g], g,
                 manifest['groove_cat_counts'][g])

  # Persist manifest so the data provider uses the same ID assignment.
  manifest_path = os.path.join(FLAGS.output_dir, 'basswave_manifest.json')
  import json
  with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
  logging.info('Wrote manifest: %s', manifest_path)

  # Eval split: deterministic via file hash so re-runs are stable.
  rng = np.random.RandomState(0)
  perm = rng.permutation(len(records))
  n_eval = int(len(records) * FLAGS.eval_split_fraction)
  eval_ids = set(perm[:n_eval])
  train_records = [r for i, r in enumerate(records) if i not in eval_ids]
  eval_records = [r for i, r in enumerate(records) if i in eval_ids]
  logging.info('Train files: %d, Eval files: %d',
               len(train_records), len(eval_records))

  hop_samples_native = FLAGS.sample_rate // FLAGS.frame_rate

  for split_name, split_records in (('train', train_records),
                                    ('eval', eval_records)):
    if not split_records:
      continue
    n_shards = max(1, FLAGS.num_shards if split_name == 'train' else 4)
    writers = []
    for s in range(n_shards):
      out_path = os.path.join(
          FLAGS.output_dir,
          f'basswave-{split_name}-{s:05d}-of-{n_shards:05d}.tfrecord')
      writers.append(tf.io.TFRecordWriter(out_path))

    n_written = 0
    import sys
    import time as _time
    split_start = _time.perf_counter()
    timing_accum = {'load': 0.0, 'resample': 0.0,
                    'features': 0.0, 'windowing': 0.0}
    n_processed = 0   # successfully processed (not skipped)

    for idx, meta in enumerate(split_records):
      if (idx + 1) % FLAGS.log_every == 0 or idx == 0:
        logging.info('  [%s] processing %d/%d: %s',
                     split_name, idx + 1, len(split_records),
                     meta['path'])
      sys.stdout.flush()
      sys.stderr.flush()

      try:
        examples, timings = process_file(
            meta, FLAGS.sample_rate, FLAGS.frame_rate,
            FLAGS.example_secs, FLAGS.hop_secs,
            FLAGS.viterbi, FLAGS.centered,
            hop_samples_native,
            min_audio_rms_db=FLAGS.min_audio_rms_db)
      except Exception as e:        # noqa
        logging.warning('Skipping %s: %s', meta['path'], e)
        continue

      shard = (meta['preset_id'] + idx) % n_shards
      for ex in examples:
        writers[shard].write(ex.SerializeToString())
        n_written += 1

      # Accumulate per-step timings.
      for k in ('load', 'resample', 'features', 'windowing'):
        timing_accum[k] += timings[k]
      n_processed += 1

      # Verbose timing for the first few files (find the bottleneck).
      if n_processed <= 5:
        total = sum(timings[k] for k in
                    ('load', 'resample', 'features', 'windowing'))
        logging.info(
            '    timing[%d]: load=%.2fs resample=%.2fs features=%.2fs '
            'windowing=%.2fs total=%.2fs (%d windows)',
            n_processed, timings['load'], timings['resample'],
            timings['features'], timings['windowing'], total,
            timings['n_windows'])

      # Periodic ETA.
      if (idx + 1) % 500 == 0:
        for w in writers:
          w.flush()
        elapsed = _time.perf_counter() - split_start
        per_file = elapsed / max(1, n_processed)
        remaining = (len(split_records) - (idx + 1)) * per_file
        avg = {k: v / max(1, n_processed) for k, v in timing_accum.items()}
        logging.info(
            '  [%s] %d/%d files (%d windows). %.2fs/file avg '
            '(load=%.2f, resample=%.2f, features=%.2f, windowing=%.2f). '
            'Elapsed=%.0fs, ETA=%.0fs (%.1fh).',
            split_name, idx + 1, len(split_records), n_written, per_file,
            avg['load'], avg['resample'], avg['features'], avg['windowing'],
            elapsed, remaining, remaining / 3600.0)

    for w in writers:
      w.close()
    logging.info('[%s] wrote %d windows across %d shards.',
                 split_name, n_written, n_shards)

  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)


# Note on Apache Beam:
# For a parallelised version of this pipeline, lift `process_file` into a
# beam.DoFn and use the same shard-writer pattern as
# prepare_tfrecord_lib.postprocess_pipeline. The Beam version is faster on
# 10k files but the single-process loop is sufficient for this dataset size
# (~30 min on a desktop with CREPE 'full' on GPU; ~3 hr on CPU).
