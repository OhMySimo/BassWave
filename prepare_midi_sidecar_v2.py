# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
r"""Produce MIDI-feature TFRecord shards joinable 1:1 al main BassWave TFR.

╔══════════════════════════════════════════════════════════════════════════╗
║ DIFFERENZA vs prepare_midi_sidecar.py (v1)                                ║
╚══════════════════════════════════════════════════════════════════════════╝

v1 riproduceva indipendentemente la pipeline di prepare_basswave: walk del
filesystem, build_records, train/eval split via RandomState(0), shard
assignment via (preset_id + idx) % n_shards. Per costruzione GARANTIVA che
shard sidecar combaciasse 1:1 con shard main? NO. Bastava una diversa
enumerazione di os.walk, una differenza di `len(records)` tra i due script,
o un file scartato in uno dei due, e gli indici divergevano. Risultato:
audio del file X accoppiato con MIDI del file Y nel join sidecar.

v2 (questo file) PIVOTA il prep: invece di re-walkare il filesystem, ITERA
direttamente i record degli shard main del TFR esistente. Per ogni main
record letto, recupera file_hash, lookup il MIDI via metadata.jsonl,
calcola le feature per la finestra corretta, e scrive nel sidecar shard
parallelo. L'ordine all'interno dello shard sidecar è uguale a quello del
main shard PER COSTRUZIONE.

Hash → MIDI:
  file_hash = int(md5(flac_path)[:15], 16)  (stessa formula di prepare_basswave)
  metadata.jsonl fornisce flac_path → midi_path
  Quindi hash_to_midi[fhash] = midi_path
  (Costruito una volta all'inizio.)

Window-index inference:
  prepare_basswave scrive tutte le finestre di un file consecutivamente nello
  stesso shard. Quindi mentre iteriamo il main shard, possiamo dedurre il
  window_idx con:
    - se file_hash == prev_file_hash: w_idx += 1
    - se diverso:                     w_idx = 0
  t0_s del finestra = w_idx * hop_secs.

MIDI cache:
  Ogni MIDI parsato una sola volta, cached. Cache cresce a ~100MB per
  l'intero dataset (20513 MIDI univoci × pochi KB ognuno). Nessuna eviction
  necessaria.

Usage:
    python prepare_midi_sidecar_v2.py \
        --main_tfr_dir   /media/simone/NVME/MidiDataset/BassWave_TFR \
        --metadata_jsonl /media/simone/NVME/MidiDataset/metadata.jsonl \
        --keymap_json    /media/simone/NVME/MidiDataset/bass_midi_keymap.json \
        --output_dir     /media/simone/NVME/MidiDataset/BassWave_TFR_MIDI \
        --override_bpm   120 \
        --offset_const_ms 6.0
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from glob import glob
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from absl import app, flags

# Env silencers.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from midi_features import (
    BassMidiKeymap, parse_midi, window_features, n_frames_for_window)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

FLAGS = flags.FLAGS

flags.DEFINE_string('main_tfr_dir', None,
                    'Dir con gli shard main TFR di prepare_basswave. '
                    'Es: /media/simone/NVME/MidiDataset/BassWave_TFR')
flags.DEFINE_string('metadata_jsonl', None,
                    'metadata.jsonl: ogni riga ha flac_path + midi_path.')
flags.DEFINE_string('keymap_json', '',
                    'bass_midi_keymap.json (opzionale).')
flags.DEFINE_string('output_dir', None,
                    'Dove scrivere gli shard sidecar.')
flags.DEFINE_string('src_prefix', 'E:/',
                    'Prefisso Windows in metadata.jsonl.')
flags.DEFINE_string('dst_prefix', '/media/simone/NVME/',
                    'Prefisso Linux corrispondente.')
flags.DEFINE_float('example_secs', 4.0, 'Window length (s). Deve matchare main.')
flags.DEFINE_float('hop_secs', 2.0, 'Hop tra finestre (s). Deve matchare main.')
flags.DEFINE_integer('frame_rate', 50, 'Frame rate features (Hz).')
flags.DEFINE_integer('sample_rate', 44100, 'SR audio.')
flags.DEFINE_bool('centered', True, 'Frame centering. Deve matchare main.')
flags.DEFINE_float('override_bpm', 120.0,
                   'BPM fisso per parsing MIDI (dataset renderizzato a 120).')
flags.DEFINE_float('offset_const_ms', 6.0,
                   'Offset costante audio-MIDI in ms.')
flags.DEFINE_integer('onset_blur', 0, 'Blur frame adiacenti per onset_mask.')
flags.DEFINE_integer('log_every', 5000, 'Log ogni N record.')
flags.DEFINE_string('splits', 'train,eval',
                    'Split da processare, comma-separated.')

flags.mark_flags_as_required(['main_tfr_dir', 'metadata_jsonl', 'output_dir'])


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _translate(p, src, dst):
  return (dst + p[len(src):]).replace('\\', '/') if p.startswith(src) else p


def _file_hash(path):
  """Same as prepare_basswave."""
  return int(hashlib.md5(path.encode()).hexdigest()[:15], 16)


# Parse "basswave-train-00031-of-00064.tfrecord" → ('train', 31, 64)
_SHARD_RE = re.compile(
    r'^basswave-(train|eval)-(\d+)-of-(\d+)\.tfrecord$')


def parse_shard_name(filename):
  """Returns (split, idx, total) or None."""
  m = _SHARD_RE.match(filename)
  if not m:
    return None
  return m.group(1), int(m.group(2)), int(m.group(3))


# Schema main TFR — quello che ci serve per ricavare file_hash.
# example_secs * sr (+ hop_size se centered) per audio, etc.
# Per semplicità leggiamo l'intero example e usiamo solo file_hash + preset_id
# (audio non viene usato in sidecar prep).
def _main_features_dict(audio_length, audio_16k_length, feat_length):
  return {
      'audio': tf.io.FixedLenFeature([audio_length], dtype=tf.float32),
      'audio_16k': tf.io.FixedLenFeature([audio_16k_length], dtype=tf.float32),
      'f0_hz': tf.io.FixedLenFeature([feat_length], dtype=tf.float32),
      'f0_confidence': tf.io.FixedLenFeature([feat_length], dtype=tf.float32),
      'loudness_db': tf.io.FixedLenFeature([feat_length], dtype=tf.float32),
      'preset_id': tf.io.FixedLenFeature([1], dtype=tf.int64),
      'transpose': tf.io.FixedLenFeature([1], dtype=tf.int64),
      'groove_cat_id': tf.io.FixedLenFeature([1], dtype=tf.int64),
      'file_hash': tf.io.FixedLenFeature([1], dtype=tf.int64),
  }


def make_sidecar_example(feats, file_hash, window_idx):
  features = {
      'onset_mask': tf.train.Feature(
          float_list=tf.train.FloatList(value=feats['onset_mask'])),
      'onset_offset_samples': tf.train.Feature(
          float_list=tf.train.FloatList(value=feats['onset_offset_samples'])),
      'onset_velocity': tf.train.Feature(
          float_list=tf.train.FloatList(value=feats['onset_velocity'])),
      'offset_mask': tf.train.Feature(
          float_list=tf.train.FloatList(value=feats['offset_mask'])),
      'offset_offset_samples': tf.train.Feature(
          float_list=tf.train.FloatList(value=feats['offset_offset_samples'])),
      'silence_mask': tf.train.Feature(
          float_list=tf.train.FloatList(value=feats['silence_mask'])),
      'active_ks_bits': tf.train.Feature(
          float_list=tf.train.FloatList(value=feats['active_ks_bits'])),
      'active_note_midi': tf.train.Feature(
          float_list=tf.train.FloatList(value=feats['active_note_midi'])),
      'file_hash': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[file_hash])),
      'window_idx': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[window_idx])),
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────


def process_shard(main_shard_path: str,
                  sidecar_shard_path: str,
                  hash_to_midi: Dict[int, str],
                  midi_cache: Dict[str, tuple],
                  parse_schema: dict,
                  keymap: Optional[BassMidiKeymap],
                  offset_s: float,
                  override_bpm: float):
  """Processa un main shard → scrive il sidecar shard parallelo.

  Args:
    main_shard_path: path al main TFRecord.
    sidecar_shard_path: path dove scrivere il sidecar (verrà sovrascritto).
    hash_to_midi: file_hash → midi_path (Linux).
    midi_cache: midi_path → (notes, ks_events). Riempito on-demand.
    parse_schema: schema per parse_single_example del main.
    keymap: BassMidiKeymap o None.
    offset_s: offset costante audio-MIDI in secondi.
    override_bpm: BPM forzato (parser MIDI).

  Returns:
    dict con statistiche del shard (n_examples, n_skipped, n_files).
  """
  log.info('shard: %s', os.path.basename(main_shard_path))
  ds = tf.data.TFRecordDataset(main_shard_path)
  writer = tf.io.TFRecordWriter(sidecar_shard_path)

  stats = {'n_examples': 0, 'n_skipped': 0, 'n_files': 0, 'n_missing': 0}
  prev_fhash = None
  w_idx = -1

  for record in ds:
    ex = tf.io.parse_single_example(record, parse_schema)
    fhash = int(ex['file_hash'].numpy().item())

    # Window index: increments while same file, resets on file change.
    if fhash != prev_fhash:
      w_idx = 0
      prev_fhash = fhash
      stats['n_files'] += 1
    else:
      w_idx += 1

    midi_path = hash_to_midi.get(fhash)
    if midi_path is None:
      # File presente in main TFR ma non in metadata.jsonl. Scriviamo un
      # esempio "vuoto" (tutti zero) per non rompere il join. La testa
      # tratterà queste finestre come "no MIDI info" durante il training
      # (loss valori = 0 sui mask, neutri).
      stats['n_missing'] += 1
      empty_feats = _make_empty_features()
      out_ex = make_sidecar_example(empty_feats, fhash, w_idx)
      writer.write(out_ex.SerializeToString())
      stats['n_examples'] += 1
      continue

    # Parse MIDI (cached).
    if midi_path not in midi_cache:
      if not os.path.exists(midi_path):
        log.warning('MIDI not found: %s', midi_path)
        stats['n_skipped'] += 1
        empty_feats = _make_empty_features()
        out_ex = make_sidecar_example(empty_feats, fhash, w_idx)
        writer.write(out_ex.SerializeToString())
        stats['n_examples'] += 1
        continue
      try:
        notes, ks = parse_midi(
            midi_path, override_bpm=override_bpm, keymap=keymap)
        for n in notes:
          n.t_on += offset_s
          n.t_off += offset_s
        for ev in ks:
          ev.t_on += offset_s
          ev.t_off += offset_s
        midi_cache[midi_path] = (notes, ks)
      except Exception as e:
        log.warning('parse_midi failed on %s: %s', midi_path, e)
        stats['n_skipped'] += 1
        empty_feats = _make_empty_features()
        out_ex = make_sidecar_example(empty_feats, fhash, w_idx)
        writer.write(out_ex.SerializeToString())
        stats['n_examples'] += 1
        continue

    notes, ks = midi_cache[midi_path]
    t0_s = w_idx * FLAGS.hop_secs

    feats = window_features(
        notes,
        t0_s=t0_s,
        example_secs=FLAGS.example_secs,
        frame_rate=FLAGS.frame_rate,
        sample_rate=FLAGS.sample_rate,
        centered=FLAGS.centered,
        onset_blur=FLAGS.onset_blur,
        ks_events=ks,
    )

    out_ex = make_sidecar_example(feats, fhash, w_idx)
    writer.write(out_ex.SerializeToString())
    stats['n_examples'] += 1

  writer.close()
  return stats


def _make_empty_features():
  T = n_frames_for_window(FLAGS.example_secs, FLAGS.frame_rate, FLAGS.centered)
  zeros = np.zeros(T, dtype=np.float32)
  return {
      'onset_mask':            zeros.copy(),
      'onset_offset_samples':  zeros.copy(),
      'onset_velocity':        zeros.copy(),
      'offset_mask':           zeros.copy(),
      'offset_offset_samples': zeros.copy(),
      'silence_mask':          np.ones(T, dtype=np.float32),  # tutto silence
      'active_ks_bits':        zeros.copy(),
      'active_note_midi':      zeros.copy(),
  }


def main(_):
  os.makedirs(FLAGS.output_dir, exist_ok=True)

  # ── Keymap ───────────────────────────────────────────────────────────
  keymap = None
  if FLAGS.keymap_json and os.path.exists(FLAGS.keymap_json):
    keymap = BassMidiKeymap(FLAGS.keymap_json)
    log.info('Keymap loaded: %s', FLAGS.keymap_json)
  else:
    log.warning('No keymap → KS = note regolari, active_ks_bits=0.')

  # ── Build hash → midi_path mapping da metadata ───────────────────────
  log.info('Building hash_to_midi from %s', FLAGS.metadata_jsonl)
  hash_to_midi: Dict[int, str] = {}
  collisions = 0
  with open(FLAGS.metadata_jsonl) as f:
    for line in f:
      d = json.loads(line)
      flac = _translate(d['flac_path'], FLAGS.src_prefix, FLAGS.dst_prefix)
      midi = _translate(d['midi_path'], FLAGS.src_prefix, FLAGS.dst_prefix)
      h = _file_hash(flac)
      if h in hash_to_midi and hash_to_midi[h] != midi:
        collisions += 1
      hash_to_midi[h] = midi
  log.info('hash_to_midi: %d entries (%d hash collisions overwritten).',
           len(hash_to_midi), collisions)

  # ── Find main TFR shards ─────────────────────────────────────────────
  splits = [s.strip() for s in FLAGS.splits.split(',')]
  all_shards = []
  for fn in sorted(os.listdir(FLAGS.main_tfr_dir)):
    pat = parse_shard_name(fn)
    if pat is None:
      continue
    split, idx, total = pat
    if split not in splits:
      continue
    all_shards.append({
        'split': split, 'idx': idx, 'total': total,
        'main_path': os.path.join(FLAGS.main_tfr_dir, fn),
        'sidecar_name': f'basswave-{split}-midi-{idx:05d}-of-{total:05d}.tfrecord',
    })
  log.info('Found %d main shards in %s for splits %s.',
           len(all_shards), FLAGS.main_tfr_dir, splits)
  if not all_shards:
    raise RuntimeError(
        f'No matching shards in {FLAGS.main_tfr_dir}. Check --main_tfr_dir.')

  # ── Build parse schema ───────────────────────────────────────────────
  # Replica del calcolo in BassWaveTFRecordProvider.__init__:
  hop_size_native = FLAGS.sample_rate // FLAGS.frame_rate
  audio_length = int(FLAGS.example_secs * FLAGS.sample_rate)
  if FLAGS.centered:
    audio_length += hop_size_native
  audio_16k_length = int(FLAGS.example_secs * 16000)
  if FLAGS.centered:
    audio_16k_length += 16000 // FLAGS.frame_rate
  feat_length = int(FLAGS.example_secs * FLAGS.frame_rate)
  if FLAGS.centered:
    feat_length += 1
  parse_schema = _main_features_dict(audio_length, audio_16k_length, feat_length)
  log.info('Parse schema: audio=%d, audio_16k=%d, features=%d',
           audio_length, audio_16k_length, feat_length)

  # ── Offset costante ──────────────────────────────────────────────────
  offset_s = FLAGS.offset_const_ms / 1000.0
  log.info('MIDI parsing: override_bpm=%.1f, offset=+%.2fms',
           FLAGS.override_bpm, FLAGS.offset_const_ms)

  # ── Cache MIDI globale (riempita on-demand, mai svuotata) ────────────
  midi_cache: Dict[str, tuple] = {}

  # ── Process shards ───────────────────────────────────────────────────
  t0_wall = time.perf_counter()
  total_stats = {'n_examples': 0, 'n_skipped': 0, 'n_files': 0, 'n_missing': 0}
  for i, shard in enumerate(all_shards):
    sidecar_path = os.path.join(FLAGS.output_dir, shard['sidecar_name'])
    stats = process_shard(
        main_shard_path=shard['main_path'],
        sidecar_shard_path=sidecar_path,
        hash_to_midi=hash_to_midi,
        midi_cache=midi_cache,
        parse_schema=parse_schema,
        keymap=keymap,
        offset_s=offset_s,
        override_bpm=FLAGS.override_bpm,
    )
    for k, v in stats.items():
      total_stats[k] += v

    elapsed = time.perf_counter() - t0_wall
    rate = (i + 1) / elapsed
    eta = (len(all_shards) - i - 1) / rate
    log.info('  [%d/%d] %s: %d examples, %d files, %d skipped, %d missing. '
             'cache=%d MIDI. ETA %.1f min',
             i + 1, len(all_shards), shard['sidecar_name'],
             stats['n_examples'], stats['n_files'],
             stats['n_skipped'], stats['n_missing'],
             len(midi_cache), eta / 60)

  elapsed = time.perf_counter() - t0_wall
  log.info('Done in %.1f min. Total: %d examples, %d files, %d skipped, '
           '%d missing (-> empty), %d MIDI parsed.',
           elapsed / 60, total_stats['n_examples'], total_stats['n_files'],
           total_stats['n_skipped'], total_stats['n_missing'],
           len(midi_cache))


if __name__ == '__main__':
  app.run(main)
