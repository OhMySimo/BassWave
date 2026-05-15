# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Data providers for the BassWave dataset (44.1 kHz mono FLAC bass).

Dataset layout (read-only on user's machine):
  FLAC_AUG/
    <groove_cat>/                    # e.g. 000500@EZbass, 000631@EBX_Classic_Rock
      <groove_pat>/                  # e.g. 053@Straight_3#4
        <bpm-Sxxx@beats>/            # e.g. 120-S718@1#16
          Variation_NN__Tp+N__Pk<preset>.flac

Filename grammar:
  Variation_(\\d+)__Tp([+-]?\\d+)__Pk(.+)\\.flac

The provider reads TFRecords produced by `prepare_basswave.py`. Each example
contains audio (44.1 k), audio_16k (CREPE), f0_hz, f0_confidence, loudness_db,
plus metadata fields (preset_id, groove_cat_id, transpose). The metadata is
optional at training time but enables preset-balanced sampling and per-preset
metrics.

Preset / groove inventory is NOT hard-coded — `prepare_basswave.py` discovers
it by walking the dataset and writes basswave_manifest.json into the same
directory as the TFRecord shards. This provider loads that manifest at
construction. If the dataset grows (new presets, new groove categories),
re-run the prep script and the IDs will be reassigned consistently
(alphabetical sort).
"""

import json
import math
import os
from typing import Dict, List, Optional

from absl import logging
from ddsp.spectral_ops import CREPE_FRAME_SIZE
from ddsp.spectral_ops import CREPE_SAMPLE_RATE
from ddsp.spectral_ops import get_framed_lengths
from ddsp.training import data as ddsp_data
import gin
import tensorflow.compat.v2 as tf


_AUTOTUNE = tf.data.experimental.AUTOTUNE


# The preset / groove inventory is no longer hard-coded. The prep script
# (prepare_basswave.py) writes a basswave_manifest.json next to the TFRecord
# shards; the data provider reads it at construction time. If the manifest
# is missing, we fall back to an empty inventory and balance_presets is a
# no-op (the provider just streams everything unweighted).
def load_manifest(tfrecord_dir_or_pattern: str) -> Optional[Dict]:
  """Look for basswave_manifest.json in the same directory as the TFRecords.

  Accepts either a directory path or a glob pattern (in which case we strip
  the basename to get the directory).
  """
  d = (tfrecord_dir_or_pattern
       if os.path.isdir(tfrecord_dir_or_pattern)
       else os.path.dirname(tfrecord_dir_or_pattern))
  manifest_path = os.path.join(d, 'basswave_manifest.json')
  if not os.path.exists(manifest_path):
    return None
  with open(manifest_path, 'r') as f:
    return json.load(f)


def preset_to_id(preset_name: str, manifest: Optional[Dict] = None) -> int:
  """Map a preset name to its int id via the manifest. Returns -1 if absent."""
  if manifest is None:
    return -1
  return manifest.get('preset_to_id', {}).get(preset_name, -1)


@gin.register
class BassWaveTFRecordProvider(ddsp_data.TFRecordProvider):
  """Reads BassWave TFRecords at 44.1 kHz with preset/groove metadata.

  Differences from the parent TFRecordProvider:
    - Default sample_rate=44100, frame_rate=50 (matches vst_48k.gin pattern).
    - features_dict includes preset_id (int), transpose (int), groove_cat_id
      (int) — needed for stratified sampling & per-preset metrics.
    - Optional weighted sampling per preset: balances the wildly skewed
      distribution (Modern=2148 vs multi_fx1=1) by sampling each preset with
      probability proportional to 1/sqrt(count). Pass `balance_presets=True`.
  """

  def __init__(self,
               file_pattern=None,
               example_secs=4,
               sample_rate=44100,
               frame_rate=50,
               centered=True,
               balance_presets=False,
               min_preset_count=0,
               manifest_path=None,
               interleave_cycle_length=8,
               interleave_block_length=4,
               num_parallel_calls=None,
               shuffle_buffer_size=64,
               prefetch_size=2):
    """Constructor.

    Args:
      file_pattern: Glob for TFRecord shards.
      example_secs: Window length in seconds (the prep script must match).
      sample_rate: Target SR (default 44.1 kHz).
      frame_rate: Frame rate for f0/ld features (default 50 Hz).
      centered: Centered framing — must match prep script.
      balance_presets: If True, rebalances via single-pipeline rejection
        sampling (keep_prob ∝ 1/sqrt(count)). Requires a manifest.
      min_preset_count: Drop presets with fewer than this many examples.
        Requires a manifest.
      manifest_path: Optional explicit path to basswave_manifest.json.
      interleave_cycle_length: Open TFRecord files in parallel. Default 8
        (was 40; each file holds ~8 buffered examples × ~950 KB each).
      interleave_block_length: Consecutive elements per file before switch.
        Default 4.
      num_parallel_calls: Worker cap for map/interleave. None → cpu_count()//2.
      shuffle_buffer_size: In-memory shuffle buffer (elements). 0 = disabled.
        Default 64. Never use AUTOTUNE here.
      prefetch_size: Batches prefetched after batching. Overrides the
        AUTOTUNE default in DataProvider.get_batch(). Default 2.
    """
    self._sample_rate = sample_rate
    self._frame_rate = frame_rate
    self._file_pattern = file_pattern or self.default_file_pattern
    self._example_secs = example_secs
    self._centered = centered
    self._balance_presets = balance_presets
    self._min_preset_count = int(min_preset_count)

    # RAM-budget pipeline parameters.
    self._interleave_cycle_length = int(interleave_cycle_length)
    self._interleave_block_length = int(interleave_block_length)
    # Cap parallel workers: None → half the logical CPUs (avoids thread-RAM
    # explosion when cycle_length was 40 and AUTOTUNE matched it).
    _ncpu = os.cpu_count() or 4
    self._num_parallel_calls = (int(num_parallel_calls)
                                if num_parallel_calls is not None
                                else max(1, _ncpu // 2))
    self._shuffle_buffer_size = int(shuffle_buffer_size)
    self._prefetch_size = int(prefetch_size)

    # Load manifest (preset/groove inventory + counts).
    if manifest_path is None:
      self._manifest = load_manifest(self._file_pattern)
    else:
      with open(manifest_path, 'r') as f:
        self._manifest = json.load(f)
    if self._balance_presets and self._manifest is None:
      logging.warning(
          'balance_presets=True but no basswave_manifest.json found near '
          '%s — falling back to uniform sampling.', self._file_pattern)
      self._balance_presets = False
    if self._min_preset_count > 0 and self._manifest is None:
      logging.warning(
          'min_preset_count=%d but no manifest found near %s — '
          'filter is a no-op.', self._min_preset_count, self._file_pattern)
      self._min_preset_count = 0

    # Compute the set of allowed preset IDs once at construction.
    self._allowed_preset_ids = self._compute_allowed_preset_ids()

    # Audio length: 4 sec * SR (+ 1 frame for center padding, like vst_48k.gin).
    self._hop_size_native = sample_rate // frame_rate  # 882 at 44.1k/50
    self._audio_length = example_secs * sample_rate
    if centered:
      self._audio_length += self._hop_size_native

    # CREPE-rate audio (always 16 kHz).
    self._audio_16k_length = example_secs * CREPE_SAMPLE_RATE
    if centered:
      self._audio_16k_length += CREPE_SAMPLE_RATE // frame_rate

    self._feature_length = self.get_feature_length(centered)

  @property
  def default_file_pattern(self):
    return os.environ.get(
        'BASSWAVE_TFRECORD_PATTERN',
        '/media/simone/NVME/MidiDataset/BassWave_TFR/basswave-train-*.tfrecord')

  def get_feature_length(self, centered):
    n = int(self._example_secs * self._frame_rate)
    return n + (1 if centered else 0)


  def _compute_allowed_preset_ids(self):
    """Build the list of preset IDs that pass the min_count filter.

    Returns None when no filter is active (manifest absent or
    min_preset_count == 0); otherwise returns a sorted list of allowed IDs.
    Logs which presets are kept vs dropped at construction time.
    """
    if self._manifest is None or self._min_preset_count <= 0:
      return None
    p2i = self._manifest['preset_to_id']
    counts = self._manifest['preset_counts']
    allowed = sorted(
        p2i[name] for name, c in counts.items()
        if c >= self._min_preset_count)
    dropped = sorted(
        p2i[name] for name, c in counts.items()
        if c < self._min_preset_count)
    if dropped:
      dropped_names = [
          name for name, c in counts.items()
          if c < self._min_preset_count]
      logging.info(
          'min_preset_count=%d: dropping %d presets (%s)',
          self._min_preset_count, len(dropped),
          ', '.join(f'{n}={counts[n]}' for n in dropped_names))
    return allowed

  def _filter_by_preset(self, ds):
    """Apply the min_preset_count filter to a tf.data.Dataset, if active."""
    if self._allowed_preset_ids is None:
      return ds
    allowed_t = tf.constant(self._allowed_preset_ids, dtype=tf.int64)
    return ds.filter(
        lambda ex: tf.reduce_any(tf.equal(ex['preset_id'], allowed_t)))

  @property
  def features_dict(self):
    """TFRecord schema: audio + features + metadata."""
    return {
        # Audio
        'audio': tf.io.FixedLenFeature(
            [self._audio_length], dtype=tf.float32),
        'audio_16k': tf.io.FixedLenFeature(
            [self._audio_16k_length], dtype=tf.float32),
        # Conditioning features (precomputed offline by prep script)
        'f0_hz': tf.io.FixedLenFeature(
            [self._feature_length], dtype=tf.float32),
        'f0_confidence': tf.io.FixedLenFeature(
            [self._feature_length], dtype=tf.float32),
        'loudness_db': tf.io.FixedLenFeature(
            [self._feature_length], dtype=tf.float32),
        # Metadata (scalar)
        'preset_id': tf.io.FixedLenFeature([1], dtype=tf.int64),
        'transpose': tf.io.FixedLenFeature([1], dtype=tf.int64),
        'groove_cat_id': tf.io.FixedLenFeature([1], dtype=tf.int64),
        # File hash for traceability (prep script writes filename hash here).
        'file_hash': tf.io.FixedLenFeature([1], dtype=tf.int64),
    }

  def _build_base_pipeline(self, shuffle: bool) -> tf.data.Dataset:
    """Single interleave pipeline shared by both balanced and unbalanced paths.

    Uses the RAM-budget parameters set at construction time:
      - cycle_length  : number of files open simultaneously (default 8, was 40)
      - block_length  : consecutive reads per file (improves I/O locality)
      - num_parallel_calls: worker cap (avoids thread-count × buffer-size RAM)
    """
    def parse_tfexample(record):
      ex = tf.io.parse_single_example(record, self.features_dict)
      for k in ('preset_id', 'transpose', 'groove_cat_id', 'file_hash'):
        ex[k] = tf.squeeze(ex[k])
      return ex

    filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=shuffle)
    ds = filenames.interleave(
        tf.data.TFRecordDataset,
        cycle_length=self._interleave_cycle_length,
        block_length=self._interleave_block_length,
        num_parallel_calls=self._num_parallel_calls)
    ds = ds.map(parse_tfexample,
                num_parallel_calls=self._num_parallel_calls)
    return ds

  def _build_rejection_weights(self) -> tf.Tensor:
    """Per-preset keep probability for rejection sampling.

    Strategy: weight_i = 1/sqrt(count_i).  Normalise so the preset with the
    *highest* weight (smallest count) has keep_prob = 1.0.  All larger presets
    are stochastically downsampled.  This is O(1) RAM: a single float32 vector
    indexed by preset_id.

    Returns:
      keep_probs: float32 tensor of shape [max_preset_id + 1].  Preset IDs
        absent from the manifest (or filtered by min_preset_count) get 0.0.
    """
    p2i = self._manifest['preset_to_id']
    counts = self._manifest['preset_counts']
    max_id = max(p2i.values()) + 1
    keep = [0.0] * max_id
    for name, pid in p2i.items():
      count = counts.get(name, 0)
      if count == 0:
        continue
      if (self._allowed_preset_ids is not None
          and pid not in self._allowed_preset_ids):
        continue
      keep[pid] = 1.0 / math.sqrt(max(count, 1))
    max_w = max(keep) if any(k > 0 for k in keep) else 1.0
    keep = [k / max_w for k in keep]
    return tf.constant(keep, dtype=tf.float32)

  def get_dataset(self, shuffle=True):
    """Returns a tf.data.Dataset, optionally rebalanced per preset.

    Balanced path uses *rejection sampling* on a **single** base pipeline.
    The previous implementation created one independent tf.data pipeline per
    preset (N copies of cycle_length=40 interleave), multiplying RAM by N.
    Rejection sampling uses O(1) extra RAM: a float32 lookup table of keep
    probabilities, and a single stochastic filter on the shared stream.
    """
    ds = self._build_base_pipeline(shuffle)
    ds = self._filter_by_preset(ds)  # min_preset_count hard filter

    if self._balance_presets:
      # Build per-preset keep probability table at graph construction time.
      keep_probs = self._build_rejection_weights()   # shape [max_id]
      max_id = keep_probs.shape[0]

      def _keep(ex):
        pid = tf.cast(ex['preset_id'], tf.int32)
        # Clamp to valid range in case of unseen preset IDs in new data.
        pid = tf.clip_by_value(pid, 0, max_id - 1)
        prob = keep_probs[pid]
        return tf.random.uniform(()) < prob

      ds = ds.filter(_keep)
      logging.info(
          'balance_presets: rejection sampling enabled. '
          'keep_probs table size: %d entries.', max_id)

    if self._shuffle_buffer_size > 0:
      ds = ds.shuffle(buffer_size=self._shuffle_buffer_size,
                      reshuffle_each_iteration=True)
    return ds

  def get_batch(self, batch_size, shuffle=True, repeats=-1,
                drop_remainder=True):
    """Wraps DataProvider.get_batch() to enforce a bounded prefetch.

    The base class uses prefetch(AUTOTUNE), which grows unboundedly over long
    training runs and is the primary driver of the OOM-killer firing at ~5000
    steps.  This override caps prefetch to self._prefetch_size (default 2).
    """
    dataset = self.get_dataset(shuffle)
    dataset = dataset.repeat(repeats)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(buffer_size=self._prefetch_size)
    return dataset
    
# ============================================================================
# SIDECAR PROVIDER PATCH — aggiungere in fondo a data_basswave.py
# ============================================================================
# BassWaveWithSidecarProvider estende BassWaveTFRecordProvider con un join
# deterministico verso i TFRecord sidecar prodotti da prepare_midi_sidecar.py.
#
# DESIGN: zip per-shard prima dello shuffle.
# ─────────────────────────────────────────
# Main shards  : basswave-train-00001-of-00064.tfrecord
# Sidecar shards: basswave-train-midi-00001-of-00064.tfrecord
# Le due liste vengono ordinate in modo identico (sorted), accoppiate 1-a-1,
# poi per ogni coppia (main_file, sidecar_file) si fa zip di due
# TFRecordDataset *sequential* (nessun interleave interno per garantire ordine
# deterministico). I record zippati escono nell'ordine corretto perché
# prepare_midi_sidecar ha usato lo stesso shard assignment di prepare_basswave.
# Il shuffle globale avviene DOPO il join, nel buffer di shuffle ereditato.
#
# Nota: per split eval i shard sono 4 per entrambi — stessa logica.
#
# Compatibilità retroattiva: se sidecar_file_pattern è None, la classe si
# comporta esattamente come il parent BassWaveTFRecordProvider.
# ============================================================================


@gin.register
class BassWaveWithSidecarProvider(BassWaveTFRecordProvider):
  """BassWave provider con join verso sidecar MIDI features.

  Extends BassWaveTFRecordProvider joining each main TFRecord shard with its
  corresponding sidecar shard (produced by prepare_midi_sidecar.py).

  The join is deterministic: both shard lists are sorted, paired by index, and
  zipped sequentially (no interleave within a pair). Global shuffle happens
  afterward inside the inherited get_dataset() shuffle buffer.

  If `sidecar_file_pattern` is None, falls back to the parent class
  (no MIDI features — backward compatible with runs that haven't run the
  sidecar prep yet).

  New fields added to each batch element (all [T] float32):
    onset_mask, onset_offset_samples, onset_velocity,
    offset_mask, offset_offset_samples, silence_mask,
    active_ks_bits, active_note_midi
  """

  # MIDI sidecar feature lengths match the main TFRecord feature length.
  _SIDECAR_KEYS = [
      'onset_mask', 'onset_offset_samples', 'onset_velocity',
      'offset_mask', 'offset_offset_samples', 'silence_mask',
      'active_ks_bits', 'active_note_midi',
  ]

  def __init__(self, *args, sidecar_file_pattern: str = None, **kwargs):
    """
    Args:
      sidecar_file_pattern: Glob for sidecar TFRecord shards. E.g.
        '/media/.../BassWave_TFR_MIDI/basswave-train-midi-*.tfrecord'.
        If None, falls back silently to parent behaviour (no MIDI features).
      *args / **kwargs: passed to BassWaveTFRecordProvider.
    """
    super().__init__(*args, **kwargs)
    self._sidecar_pattern = sidecar_file_pattern

  @property
  def sidecar_features_dict(self):
    """Parse schema for sidecar TFRecord examples."""
    T = self._feature_length
    schema = {}
    for key in self._SIDECAR_KEYS:
      schema[key] = tf.io.FixedLenFeature([T], dtype=tf.float32)
    # Debug keys written by prepare_midi_sidecar (ignored at training time
    # but needed so parse_single_example doesn't error on unexpected fields).
    schema['file_hash']  = tf.io.FixedLenFeature([1], dtype=tf.int64)
    schema['window_idx'] = tf.io.FixedLenFeature([1], dtype=tf.int64)
    return schema

  def _build_base_pipeline(self, shuffle: bool) -> tf.data.Dataset:
    """Override: zip main + sidecar per-shard, then optionally shuffle pairs."""
    if self._sidecar_pattern is None:
      # No sidecar — delegate to parent unchanged.
      return super()._build_base_pipeline(shuffle)

    # ── Resolve and sort both shard lists ────────────────────────────────
    main_files    = sorted(tf.io.gfile.glob(self._file_pattern))
    sidecar_files = sorted(tf.io.gfile.glob(self._sidecar_pattern))

    if len(main_files) == 0:
      raise RuntimeError(
          f'No main TFRecord files found: {self._file_pattern}')
    if len(sidecar_files) == 0:
      logging.warning(
          'No sidecar TFRecord files found: %s — '
          'falling back to parent provider (no MIDI features).',
          self._sidecar_pattern)
      return super()._build_base_pipeline(shuffle)
    if len(main_files) != len(sidecar_files):
      raise RuntimeError(
          f'Main / sidecar shard count mismatch: '
          f'{len(main_files)} main vs {len(sidecar_files)} sidecar. '
          f'Re-run prepare_midi_sidecar.py with the same --num_shards.')

    # ── Build dataset of (main_path, sidecar_path) pairs ─────────────────
    main_t    = tf.constant(main_files,    dtype=tf.string)
    sidecar_t = tf.constant(sidecar_files, dtype=tf.string)
    pair_ds   = tf.data.Dataset.from_tensor_slices((main_t, sidecar_t))

    if shuffle:
      # Shuffle at the shard level first (O(n_shards) RAM).
      pair_ds = pair_ds.shuffle(
          buffer_size=len(main_files),
          reshuffle_each_iteration=True)

    # ── For each pair: zip two *sequential* TFRecordDatasets ─────────────
    # Use flat_map (not interleave) to preserve per-pair ordering.
    # deterministic=True is the default but we set it explicitly.
    def _read_pair(main_f, sidecar_f):
      main_ds    = tf.data.TFRecordDataset(main_f)
      sidecar_ds = tf.data.TFRecordDataset(sidecar_f)
      return tf.data.Dataset.zip((main_ds, sidecar_ds))

    zipped_ds = pair_ds.flat_map(_read_pair)

    # ── Parse both halves and merge ───────────────────────────────────────
    main_schema    = self.features_dict
    sidecar_schema = self.sidecar_features_dict

    def _parse_pair(main_rec, sidecar_rec):
      main_ex = tf.io.parse_single_example(main_rec, main_schema)
      sc_ex   = tf.io.parse_single_example(sidecar_rec, sidecar_schema)
      # Squeeze scalar metadata keys (mirrors parent _build_base_pipeline).
      for k in ('preset_id', 'transpose', 'groove_cat_id', 'file_hash'):
        main_ex[k] = tf.squeeze(main_ex[k])
      # Merge sidecar features into main example.
      # Drop sidecar debug keys (file_hash already in main, window_idx unused).
      for k in self._SIDECAR_KEYS:
        main_ex[k] = sc_ex[k]
      return main_ex

    ds = zipped_ds.map(
        _parse_pair,
        num_parallel_calls=self._num_parallel_calls,
        deterministic=True)   # must stay True to preserve zip order

    return ds
