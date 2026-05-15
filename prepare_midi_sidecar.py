# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
r"""Produce MIDI-feature TFRecord shards che mirroring il BassWave main prep.

Per ogni FLAC processato da prepare_basswave, questo script produce una riga
corrispondente in un "sidecar" TFRecord con 8 feature temporali derivate dal
MIDI. I sidecar hanno la STESSA struttura di shard (stesso shard-key, stesso
ordine di file, stesso train/eval split) del main TFR, cosi' il data provider
puo' fare `tf.data.Dataset.zip()` prima dello shuffle e ottenere join 1:1.

Shard key replicato esattamente da prepare_basswave:
    shard = (preset_id + file_idx_within_split) % n_shards

Feature scritte per ogni window:
    onset_mask            [T] float32  (1.0 = frame con onset)
    onset_offset_samples  [T] float32  (residuo sub-frame in [-441, +441])
    onset_velocity        [T] float32  (MIDI velocity 0-127 / 127.0)
    offset_mask           [T] float32  (1.0 = frame con note-off)
    offset_offset_samples [T] float32
    silence_mask          [T] float32  (1.0 = nessuna nota attiva)
    active_ks_bits        [T] float32  (bitmask intera dei KS attivi come float;
                                        bit i = KS index i attivo, i=0..11.
                                        Richiede --keymap_json.)
    active_note_midi      [T] float32  (MIDI pitch nota attiva, 0=silenzio;
                                        range 21-64. Richiede --keymap_json.)
    file_hash             int64        (identico al main TFR per debug)
    window_idx            int64        (0-based per debug)

Parametri MIDI fissati dal dataset BassWave:
    override_bpm=120.0  (renderer normalizza a 120 BPM)
    offset_const=+0.006 s (+6ms, mediana misurata su 26575 file)

Integrazione KeyMap (--keymap_json):
    Quando --keymap_json e' fornito, i MIDI event con pitch 9-20 (key switch
    range A-1..G#0 del plugin bass) vengono estratti come KeySwitchEvent e
    NON contribuiscono a onset_mask/offset_mask/silence_mask. Questo elimina
    i falsi onset che si verificavano prima quando i KS venivano trattati come
    note regolari. Le feature active_ks_bits e active_note_midi vengono
    popolate solo con keymap attivo; senza keymap valgono 0 per tutti i frame.

Usage:
    python prepare_midi_sidecar.py \
        --input_root   /media/simone/NVME/MidiDataset/FLAC_AUG \
        --midi_root    /media/simone/NVME/MidiDataset/MIDI_AUG \
        --metadata_jsonl /media/simone/NVME/MidiDataset/metadata.jsonl \
        --manifest_json  /media/simone/NVME/MidiDataset/BassWave_TFR/basswave_manifest.json \
        --keymap_json    /media/simone/NVME/MidiDataset/bass_midi_keymap.json \
        --output_dir   /media/simone/NVME/MidiDataset/BassWave_TFR_MIDI \
        --num_shards   64 \
        --eval_split_fraction 0.05 \
        --skip_list    ./midi_offset_full_silent_files.txt
"""

import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from absl import app, flags

# --------------------------------------------------------------------------
# Env silencers.
# --------------------------------------------------------------------------
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

# --------------------------------------------------------------------------
# Import MIDI parser from same dir.
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from midi_features import (
    BassMidiKeymap, parse_midi, windowed_features_for_file)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string('input_root', None,
                    'Root dir delle FLAC (FLAC_AUG).')
flags.DEFINE_string('midi_root', None,
                    'Root dir dei MIDI (MIDI_AUG).')
flags.DEFINE_string('metadata_jsonl', None,
                    'metadata.jsonl con campo midi_path per ogni FLAC.')
flags.DEFINE_string('manifest_json', None,
                    'basswave_manifest.json scritto da prepare_basswave.')
flags.DEFINE_string('keymap_json', '',
                    'Path a bass_midi_keymap.json. '
                    'Se fornito, i key switch (MIDI 9-20) vengono separati '
                    'dalle note regolari (MIDI 21-64): niente falsi onset sui '
                    'KS, e le feature active_ks_bits / active_note_midi '
                    'vengono popolate. Senza keymap queste due feature '
                    'valgono 0 per tutti i frame.')
flags.DEFINE_string('output_dir', None,
                    'Dove scrivere i sidecar TFRecord.')
flags.DEFINE_integer('num_shards', 64,
                     'Numero shard train (eval usa sempre 4).')
flags.DEFINE_float('eval_split_fraction', 0.05,
                   'Stessa frazione del main prep (default 0.05).')
flags.DEFINE_string('skip_list', '',
                    'File txt con path FLAC da saltare (uno per riga).')
flags.DEFINE_string('src_prefix', 'E:/',
                    'Prefisso Windows nei path midi_path di metadata.jsonl.')
flags.DEFINE_string('dst_prefix', '/media/simone/NVME/',
                    'Prefisso Linux corrispondente.')
flags.DEFINE_float('example_secs', 4.0, 'Stessa del main prep.')
flags.DEFINE_float('hop_secs', 2.0, 'Stessa del main prep.')
flags.DEFINE_integer('frame_rate', 50, 'Stessa del main prep.')
flags.DEFINE_integer('sample_rate', 44100, 'SR audio del main prep.')
flags.DEFINE_float('override_bpm', 120.0,
                   'BPM fisso per parsing MIDI (renderer normalizza a 120).')
flags.DEFINE_float('offset_const_ms', 6.0,
                   'Offset costante audio-MIDI in ms (mediana su 26575 file).')
flags.DEFINE_integer('onset_blur', 0,
                     'Blur su frame adiacenti per onset_mask. 0=off.')
flags.DEFINE_integer('log_every', 500, 'Log ogni N file.')

flags.mark_flags_as_required([
    'input_root', 'midi_root', 'metadata_jsonl',
    'manifest_json', 'output_dir'])


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------


def _translate(p: str, src: str, dst: str) -> str:
  return (dst + p[len(src):]).replace('\\', '/') if p.startswith(src) else p


def _file_hash(path: str) -> int:
  return int(hashlib.md5(path.encode()).hexdigest()[:15], 16)


def _load_skip_set(skip_list_path: str) -> set:
  if not skip_list_path or not os.path.exists(skip_list_path):
    return set()
  with open(skip_list_path) as f:
    return {line.strip() for line in f if line.strip()}


def walk_dataset_paths(root: str) -> List[str]:
  paths = []
  for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
      if fn.endswith('.flac'):
        paths.append(os.path.join(dirpath, fn))
  return paths


def build_records(flac_paths: List[str], manifest: dict,
                  flac_to_midi: Dict[str, str]) -> List[dict]:
  """Costruisce lista record con preset_id, file_hash, midi_path."""
  p2i = manifest['preset_to_id']
  g2i = manifest['groove_cat_to_id']
  records = []
  for path in flac_paths:
    if path not in flac_to_midi:
      continue
    fname = os.path.basename(path)
    pk_start = fname.find('__Pk')
    if pk_start < 0:
      continue
    preset = fname[pk_start + 4:].replace('.flac', '')
    if preset not in p2i:
      continue
    parts = path.replace(os.sep, '/').split('/')
    try:
      groove_cat = parts[-4]
    except IndexError:
      groove_cat = 'unknown'
    if groove_cat not in g2i:
      groove_cat = sorted(g2i.keys())[0]
    records.append({
        'path': path,
        'midi_path': flac_to_midi[path],
        'preset': preset,
        'preset_id': p2i[preset],
        'groove_cat': groove_cat,
        'groove_cat_id': g2i[groove_cat],
        'file_hash': _file_hash(path),
    })
  return records


def make_sidecar_example(feats: dict, file_hash: int,
                         window_idx: int) -> tf.train.Example:
  """Serializza le 8 feature MIDI + chiavi di debug in un tf.train.Example.

  Feature scritte:
    6 feature originali (onset_mask, onset_offset_samples, onset_velocity,
    offset_mask, offset_offset_samples, silence_mask) + 2 nuove dalla keymap
    (active_ks_bits, active_note_midi) + 2 chiavi debug (file_hash, window_idx).

  active_ks_bits e active_note_midi sono 0 per tutti i frame se il keymap non
  e' stato fornito (parse_midi chiamato senza keymap=...).
  """
  features = {
      # --- Feature originali (6) ---
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
      # --- Feature keymap (2) ---
      # active_ks_bits: bitmask intera (come float) dei KS attivi nel frame.
      #   bit i = KS con indice i (0=KS_POLYPHONY/midi9 ... 11=KS_LEGATO/midi20).
      #   Decode esempio: bit_i_attivo = int(v) & (1 << i)
      #   Senza keymap: array di zeri.
      'active_ks_bits': tf.train.Feature(
          float_list=tf.train.FloatList(value=feats['active_ks_bits'])),
      # active_note_midi: pitch MIDI della nota attiva (0=silenzio, 21-64 altrimenti).
      #   Senza keymap: array di zeri.
      'active_note_midi': tf.train.Feature(
          float_list=tf.train.FloatList(value=feats['active_note_midi'])),
      # --- Debug keys ---
      'file_hash': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[file_hash])),
      'window_idx': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[window_idx])),
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------


def main(_):
  os.makedirs(FLAGS.output_dir, exist_ok=True)

  # ── Carica keymap (opzionale) ──────────────────────────────────────────
  keymap: Optional[BassMidiKeymap] = None
  if FLAGS.keymap_json and os.path.exists(FLAGS.keymap_json):
    keymap = BassMidiKeymap(FLAGS.keymap_json)
    log.info('Keymap loaded: %s  (N_KS=%d, N_NOTES=%d)',
             FLAGS.keymap_json, keymap.N_KS, keymap.N_NOTES)
    log.info('  KS range   : MIDI %d-%d  (key switch, esclusi da onset_mask)',
             keymap.KS_MIDI_MIN, keymap.KS_MIDI_MAX)
    log.info('  Note range : MIDI %d-%d  (note regolari, A0-E4)',
             keymap.NOTE_MIDI_MIN, keymap.NOTE_MIDI_MAX)
  else:
    log.warning('--keymap_json non fornito o file non trovato. '
                'I key switch verranno trattati come note regolari '
                '(falsi onset possibili). active_ks_bits e active_note_midi '
                'saranno 0 per tutti i frame.')

  # ── Load manifest ──────────────────────────────────────────────────────
  log.info('Loading manifest: %s', FLAGS.manifest_json)
  with open(FLAGS.manifest_json) as f:
    manifest = json.load(f)

  # ── Build FLAC->MIDI mapping ───────────────────────────────────────────
  log.info('Reading metadata: %s', FLAGS.metadata_jsonl)
  flac_to_midi: Dict[str, str] = {}
  with open(FLAGS.metadata_jsonl) as f:
    for line in f:
      d = json.loads(line)
      flac = _translate(d['flac_path'], FLAGS.src_prefix, FLAGS.dst_prefix)
      midi = _translate(d['midi_path'], FLAGS.src_prefix, FLAGS.dst_prefix)
      flac_to_midi[flac] = midi

  # ── Walk FLAC_AUG ──────────────────────────────────────────────────────
  log.info('Walking: %s', FLAGS.input_root)
  flac_paths = walk_dataset_paths(FLAGS.input_root)
  log.info('Found %d FLAC files.', len(flac_paths))

  records = build_records(flac_paths, manifest, flac_to_midi)
  log.info('%d records with MIDI mapping.', len(records))

  skip_set = _load_skip_set(FLAGS.skip_list)
  if skip_set:
    log.info('Skip list: %d files.', len(skip_set))

  # ── Train/eval split ───────────────────────────────────────────────────
  rng = np.random.RandomState(0)
  perm = rng.permutation(len(records))
  n_eval = int(len(records) * FLAGS.eval_split_fraction)
  eval_ids = set(perm[:n_eval])
  train_records = [r for i, r in enumerate(records) if i not in eval_ids]
  eval_records  = [r for i, r in enumerate(records) if i in eval_ids]
  log.info('Train: %d  Eval: %d', len(train_records), len(eval_records))

  # ── Constants ──────────────────────────────────────────────────────────
  offset_const_s = FLAGS.offset_const_ms / 1000.0

  # ── Process each split ────────────────────────────────────────────────
  for split_name, split_records in (('train', train_records),
                                    ('eval', eval_records)):
    if not split_records:
      continue
    n_shards = FLAGS.num_shards if split_name == 'train' else 4
    writers = []
    for s in range(n_shards):
      p = os.path.join(
          FLAGS.output_dir,
          f'basswave-{split_name}-midi-{s:05d}-of-{n_shards:05d}.tfrecord')
      writers.append(tf.io.TFRecordWriter(p))

    n_written = n_skipped = n_err = 0
    t0_wall = time.perf_counter()

    for idx, rec in enumerate(split_records):
      flac_path = rec['path']
      midi_path = rec['midi_path']

      if flac_path in skip_set or not os.path.exists(flac_path):
        n_skipped += 1
        continue
      if not os.path.exists(midi_path):
        log.warning('MIDI not found, skipping: %s', midi_path)
        n_skipped += 1
        continue

      try:
        import soundfile as sf
        audio, sr = sf.read(flac_path, dtype='float32', always_2d=False)
        if audio.ndim > 1:
          audio = audio.mean(axis=1)
        dur_s = len(audio) / sr

        # ── Parse MIDI con keymap ──────────────────────────────────────
        # Con keymap: notes=pitch 21-64, ks_events=pitch 9-20 (separati).
        # Senza keymap: notes=tutti i pitch, ks_events=[].
        notes, ks_events = parse_midi(
            midi_path,
            override_bpm=FLAGS.override_bpm,
            keymap=keymap)

        # Applica offset costante audio-MIDI a note E key switch.
        for n in notes:
          n.t_on  += offset_const_s
          n.t_off += offset_const_s
        for ev in ks_events:
          ev.t_on  += offset_const_s
          ev.t_off += offset_const_s

        # ── Feature per finestra ───────────────────────────────────────
        win_feats = windowed_features_for_file(
            notes,
            ks_events=ks_events,
            file_duration_s=dur_s,
            example_secs=FLAGS.example_secs,
            hop_secs=FLAGS.hop_secs,
            frame_rate=FLAGS.frame_rate,
            sample_rate=sr,
            centered=True,
            onset_blur=FLAGS.onset_blur)

        if not win_feats:
          n_skipped += 1
          continue

        shard = (rec['preset_id'] + idx) % n_shards
        for w_idx, feats in enumerate(win_feats):
          ex = make_sidecar_example(feats, rec['file_hash'], w_idx)
          writers[shard].write(ex.SerializeToString())
          n_written += 1

      except Exception as e:
        log.warning('Error on %s: %s', os.path.basename(flac_path), e)
        n_err += 1
        continue

      if (idx + 1) % FLAGS.log_every == 0:
        elapsed = time.perf_counter() - t0_wall
        rate = (idx + 1) / elapsed
        eta = (len(split_records) - idx - 1) / rate
        log.info('[%s] %d/%d  %.1f f/s  ETA %.1f min  '
                 'written=%d skipped=%d err=%d',
                 split_name, idx + 1, len(split_records),
                 rate, eta / 60, n_written, n_skipped, n_err)

    for w in writers:
      w.close()
    elapsed = time.perf_counter() - t0_wall
    log.info('[%s] done. written=%d skipped=%d err=%d in %.1f min',
             split_name, n_written, n_skipped, n_err, elapsed / 60)

  log.info('Sidecar prep complete. Output: %s', FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
