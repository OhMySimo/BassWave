# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Checkpoint surgery: expand decoder from 3 input_keys → 4 input_keys.

Phase 1 ckpt has RnnFcDecoder(input_keys=('ld_scaled', 'f0_scaled', 'z')).
Phase 2 needs RnnFcDecoder(input_keys=('ld_scaled', 'f0_scaled', 'z',
'midi_cond')).

Three shape changes when going 3 → 4 input keys:
  • input_stacks      : len 3 → 4 (4th stack is new, init random)
  • rnn.kernel        : (1536, 3*units) → (2048, 3*units)
  • out_stack[0].kernel: (2048, 512)    → (2560, 512)

Zero-init the new axis-0 rows so at step 0 the new decoder produces output
IDENTICAL to Phase 1 (midi_cond × 0 = 0). No spectral_loss shock.

────────────────────────────────────────────────────────────────────────────
LESSONS LEARNED (v1, v2 failures):

v1: used dict.get(name) with raw variable names. When two model instances of
    the same Keras subclass exist in one process, TF auto-numbers them (old =
    `denoising_autoencoder/...`, new = `denoising_autoencoder_1/...`). NO
    variable matched. Saved checkpoint had only preprocessor (6 vars), all
    other modules were random-init → catastrophic silent failure.

v2: tried name normalization (strip trailing _NN from each segment). FIXED
    the wrapper suffix mismatch BUT introduced a NEW collision: within an
    FcStack, layers are named `dense`, `dense_1`, `dense_2` (positionally
    meaningful). Normalization makes them all collide on `dense`. Dict only
    keeps the LAST → 2/3 of FcStack weights silently wrong.

v3 (this file): POSITIONAL weight matching. Walk old.weights and new.weights
    in parallel as PARALLEL LISTS. Order of variables in `model.weights` is
    deterministic per-model when the architecture is structurally identical
    on the matched submodules. Differences happen ONLY at:
      • input_stacks (count differs by 1: handle explicitly)
      • rnn kernel (axis-0 mismatch: handle with pad)
      • out_stack first dense kernel (axis-0 mismatch: handle with pad)
    Outside the decoder, modules are bit-identical → positional match is safe.

OTHER FIXES vs v1:
  • Gin path: auto-detect gin/ from ddsp install, error out if missing.
  • Sanity guard: require ≥80 % of new vars transferred. Refuse otherwise.
────────────────────────────────────────────────────────────────────────────

Usage
-----
    python expand_decoder_for_midi_cond.py \
        --phase1_dir /media/simone/NVME/runs/basswave_v4_phase1 \
        --output_dir /media/simone/NVME/runs/basswave_v4_phase2 \
        --phase1_gin papers/basswave/phase1_patched.gin \
        --phase2_gin papers/basswave/basswave_midi_head_phase2.gin
"""
from __future__ import annotations

import argparse
import os
import sys

import gin
import numpy as np
import tensorflow.compat.v2 as tf

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


# ─────────────────────────────────────────────────────────────────────────────
# Color
# ─────────────────────────────────────────────────────────────────────────────

def _green(s):  return f'\033[92m{s}\033[0m'
def _red(s):    return f'\033[91m{s}\033[0m'
def _yellow(s): return f'\033[93m{s}\033[0m'


# ─────────────────────────────────────────────────────────────────────────────
# Gin search path
# ─────────────────────────────────────────────────────────────────────────────

def find_gin_dir() -> str:
  candidates = []
  try:
    import ddsp.training
    candidates.append(os.path.join(
        os.path.dirname(ddsp.training.__file__), 'gin'))
  except ImportError:
    pass
  for start in [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
    cur = start
    for _ in range(6):
      candidates.append(os.path.join(cur, 'gin'))
      candidates.append(os.path.join(cur, 'ddsp', 'training', 'gin'))
      cur = os.path.dirname(cur)
      if cur in ('/', ''):
        break
  for c in candidates:
    if os.path.isdir(os.path.join(c, 'papers', 'basswave')):
      return c
  raise RuntimeError('Could not find gin/ dir. Searched: ' +
                     ', '.join(candidates))


# ─────────────────────────────────────────────────────────────────────────────
# Per-variable transfer
# ─────────────────────────────────────────────────────────────────────────────

def transfer_variable(src_var: tf.Variable,
                      dst_var: tf.Variable) -> str:
  """Copy src→dst with zero-pad on a single mismatched axis."""
  src_shape = src_var.shape.as_list()
  dst_shape = dst_var.shape.as_list()

  if src_shape == dst_shape:
    dst_var.assign(src_var)
    return 'exact'

  if len(src_shape) != len(dst_shape):
    return f'skip(rank_mismatch src={src_shape} dst={dst_shape})'

  diff_axes = [i for i, (s, d) in enumerate(zip(src_shape, dst_shape))
               if s != d]
  if len(diff_axes) != 1:
    return f'skip(multi_axis src={src_shape} dst={dst_shape})'

  axis = diff_axes[0]
  pad = dst_shape[axis] - src_shape[axis]
  if pad < 0:
    return f'skip(dst_smaller src={src_shape} dst={dst_shape})'

  src_arr = src_var.numpy()
  pad_config = [(0, 0)] * len(src_shape)
  pad_config[axis] = (0, pad)
  padded = np.pad(src_arr, pad_config, mode='constant', constant_values=0.0)
  dst_var.assign(padded)
  return f'padded(axis={axis}, +{pad})'


def transfer_weights_positional(src_weights: list,
                                dst_weights: list,
                                label: str,
                                verbose: bool = True) -> dict:
  """Transfer weights by POSITION (zip parallel lists).

  Requires len(src_weights) == len(dst_weights). For each pair, names should
  end with the same suffix (e.g. both end in 'kernel:0') as a sanity check.
  """
  stats = {'exact': 0, 'padded': 0, 'skip': 0, 'mismatched_suffix': 0}

  if len(src_weights) != len(dst_weights):
    print(_red(f'  {label}: weight count mismatch '
               f'src={len(src_weights)} dst={len(dst_weights)}. '
               f'Refusing positional match.'))
    stats['skip'] = len(dst_weights)
    return stats

  for src_var, dst_var in zip(src_weights, dst_weights):
    # Sanity: suffix should match (e.g. '/kernel:0' == '/kernel:0').
    src_suffix = src_var.name.rsplit('/', 1)[-1]
    dst_suffix = dst_var.name.rsplit('/', 1)[-1]
    if src_suffix != dst_suffix:
      stats['mismatched_suffix'] += 1
      if verbose:
        print(_yellow(f'    suffix mismatch: src={src_var.name} '
                      f'dst={dst_var.name} — attempting anyway'))

    result = transfer_variable(src_var, dst_var)
    if result == 'exact':
      stats['exact'] += 1
    elif result.startswith('padded'):
      stats['padded'] += 1
      if verbose:
        print(f'    {_green(result)}: {dst_var.name} '
              f'({src_var.shape} → {dst_var.shape})')
    else:
      stats['skip'] += 1
      print(f'    {_red(result)}: {dst_var.name}')

  return stats


def transfer_module(src_module: tf.keras.layers.Layer,
                    dst_module: tf.keras.layers.Layer,
                    label: str,
                    verbose: bool = True) -> dict:
  """Positional transfer of all .weights between two structurally identical
  modules."""
  return transfer_weights_positional(
      list(src_module.weights), list(dst_module.weights),
      label, verbose=verbose)


def transfer_decoder(old_dec, new_dec) -> dict:
  """RnnFcDecoder: input_stacks count changes, rnn+out_stack shapes change."""
  print('\n  decoder (special handling):')

  total = {'exact': 0, 'padded': 0, 'skip': 0, 'mismatched_suffix': 0}

  # input_stacks[0..2]: positional transfer per-stack.
  n_shared = min(len(old_dec.input_stacks), len(new_dec.input_stacks))
  for i in range(n_shared):
    sub = transfer_module(
        old_dec.input_stacks[i], new_dec.input_stacks[i],
        f'decoder.input_stacks[{i}]', verbose=False)
    for k in total:
      total[k] += sub[k]
  print(f'    input_stacks[0..{n_shared-1}]: '
        f'exact={total["exact"]} padded={total["padded"]} '
        f'skip={total["skip"]}')

  if len(new_dec.input_stacks) > n_shared:
    n_new = sum(len(s.weights) for s in new_dec.input_stacks[n_shared:])
    print(_yellow(f'    input_stacks[{n_shared}..'
                  f'{len(new_dec.input_stacks)-1}]: '
                  f'{n_new} weights left random-init '
                  f'(new midi_cond stack)'))

  # rnn: kernel shape changes (input dim grows).
  print('    rnn:')
  rnn = transfer_module(old_dec.rnn, new_dec.rnn, 'decoder.rnn', verbose=True)
  for k in total:
    total[k] += rnn[k]

  # out_stack: first Dense kernel shape changes.
  print('    out_stack:')
  out = transfer_module(old_dec.out_stack, new_dec.out_stack,
                        'decoder.out_stack', verbose=True)
  for k in total:
    total[k] += out[k]

  # dense_out: identical.
  dense_out = transfer_module(old_dec.dense_out, new_dec.dense_out,
                              'decoder.dense_out', verbose=False)
  print(f'    dense_out: exact={dense_out["exact"]} '
        f'padded={dense_out["padded"]} skip={dense_out["skip"]}')
  for k in total:
    total[k] += dense_out[k]

  return total


# ─────────────────────────────────────────────────────────────────────────────
# Gin + build
# ─────────────────────────────────────────────────────────────────────────────

def parse_gin(base_gins, overlay_gin, gin_dir):
  gin.clear_config()
  gin.add_config_file_search_path(gin_dir)
  for cfg in list(base_gins) + [overlay_gin]:
    abs_path = cfg if os.path.isabs(cfg) else os.path.join(gin_dir, cfg)
    if not os.path.isfile(abs_path):
      raise FileNotFoundError(
          f'Gin file not found: {cfg} (tried {abs_path}, gin_dir={gin_dir})')
    gin.parse_config_file(abs_path)


def build_and_forward(strategy):
  from ddsp.training.models import get_model
  with strategy.scope():
    model = get_model()

  B = 1
  dummy = {
      'audio':         tf.zeros([B, 177282], dtype=tf.float32),
      'audio_16k':     tf.zeros([B, 64320],  dtype=tf.float32),
      'f0_hz':         tf.ones ([B, 201],    dtype=tf.float32) * 60.0,
      'f0_confidence': tf.ones ([B, 201],    dtype=tf.float32) * 0.8,
      'loudness_db':   tf.ones ([B, 201],    dtype=tf.float32) * -30.0,
      'preset_id':     tf.constant([0], dtype=tf.int64),
      'transpose':     tf.constant([0], dtype=tf.int64),
      'groove_cat_id': tf.constant([0], dtype=tf.int64),
      'file_hash':     tf.constant([0], dtype=tf.int64),
      'onset_mask':            tf.zeros([B, 201], dtype=tf.float32),
      'onset_offset_samples':  tf.zeros([B, 201], dtype=tf.float32),
      'onset_velocity':        tf.zeros([B, 201], dtype=tf.float32),
      'offset_mask':           tf.zeros([B, 201], dtype=tf.float32),
      'offset_offset_samples': tf.zeros([B, 201], dtype=tf.float32),
      'silence_mask':          tf.ones ([B, 201], dtype=tf.float32),
      'active_ks_bits':        tf.zeros([B, 201], dtype=tf.float32),
      'active_note_midi':      tf.zeros([B, 201], dtype=tf.float32),
  }
  _ = model(dummy, training=False)
  return model


def find_latest_checkpoint(directory):
  latest = tf.train.latest_checkpoint(directory)
  if latest is None:
    raise FileNotFoundError(f'No checkpoint found in {directory}.')
  return latest


def extract_step_number(ckpt_prefix):
  basename = os.path.basename(ckpt_prefix)
  try:
    return int(basename.split('-')[-1])
  except (ValueError, IndexError):
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--phase1_dir', required=True)
  ap.add_argument('--output_dir', required=True)
  ap.add_argument('--phase1_gin', required=True)
  ap.add_argument('--phase2_gin', required=True)
  ap.add_argument('--gin_search_path', default=None)
  ap.add_argument('--base_gins', nargs='*',
                  default=['papers/basswave/basswave_44k.gin',
                           'papers/basswave/basswave_ram_budget.gin'])
  ap.add_argument('--dry_run', action='store_true')
  args = ap.parse_args()

  gin_dir = args.gin_search_path or find_gin_dir()
  print(f'Using gin directory: {gin_dir}')
  if not os.path.isdir(os.path.join(gin_dir, 'papers', 'basswave')):
    print(_red(f'FAIL: {gin_dir} missing papers/basswave/'))
    sys.exit(1)

  strategy = tf.distribute.get_strategy()

  # ── 1. Build OLD ──────────────────────────────────────────────────────
  print('\n' + '=' * 70)
  print('STEP 1: Building OLD model (Phase 1, 3-key decoder)')
  print('=' * 70)
  parse_gin(args.base_gins, args.phase1_gin, gin_dir)
  old_model = build_and_forward(strategy)
  print(f'  old decoder input_stacks: {len(old_model.decoder.input_stacks)}')
  print(f'  old total params: '
        f'{sum(int(np.prod(v.shape)) for v in old_model.weights) / 1e6:.2f}M')

  # ── 2. Restore OLD ────────────────────────────────────────────────────
  print('\n' + '=' * 70)
  print('STEP 2: Restoring Phase 1 checkpoint into OLD model')
  print('=' * 70)
  phase1_ckpt = find_latest_checkpoint(args.phase1_dir)
  phase1_step = extract_step_number(phase1_ckpt)
  print(f'  Phase 1 ckpt: {phase1_ckpt}  (step {phase1_step})')
  restore_ckpt = tf.train.Checkpoint(model=old_model)
  status = restore_ckpt.restore(phase1_ckpt)
  status.expect_partial()
  try:
    status.assert_existing_objects_matched()
    print(_green('  OK: all variables in old_model matched the ckpt'))
  except Exception as e:
    print(_yellow(f'  warn: not all variables matched: {e}'))

  # ── 3. Build NEW ──────────────────────────────────────────────────────
  print('\n' + '=' * 70)
  print('STEP 3: Building NEW model (Phase 2, 4-key decoder)')
  print('=' * 70)
  parse_gin(args.base_gins, args.phase2_gin, gin_dir)
  new_model = build_and_forward(strategy)
  print(f'  new decoder input_stacks: {len(new_model.decoder.input_stacks)}')
  print(f'  new total params: '
        f'{sum(int(np.prod(v.shape)) for v in new_model.weights) / 1e6:.2f}M')

  if len(new_model.decoder.input_stacks) <= len(old_model.decoder.input_stacks):
    print(_red('FAIL: new decoder has same/fewer input_stacks.'))
    sys.exit(1)

  # ── 4. Transfer (POSITIONAL) ──────────────────────────────────────────
  print('\n' + '=' * 70)
  print('STEP 4: Transferring weights (POSITIONAL matching)')
  print('=' * 70)
  total = {'exact': 0, 'padded': 0, 'skip': 0, 'mismatched_suffix': 0}

  for name in ['preprocessor', 'encoder', 'processor_group',
               'mel_extractor', 'midi_head']:
    src = getattr(old_model, name, None)
    dst = getattr(new_model, name, None)
    if src is None or dst is None:
      print(_yellow(f'  {name}: not present on both, skip'))
      continue
    if not src.weights or not dst.weights:
      print(f'  {name}: 0 weights, skip')
      continue
    print(f'\n  {name}:')
    stats = transfer_module(src, dst, name, verbose=False)
    for k in total:
      total[k] += stats[k]
    print(f'    exact={stats["exact"]} padded={stats["padded"]} '
          f'skip={stats["skip"]}')

  dec_stats = transfer_decoder(old_model.decoder, new_model.decoder)
  for k in total:
    total[k] += dec_stats[k]

  n_new = len(new_model.weights)
  n_transferred = total['exact'] + total['padded']
  ratio = n_transferred / n_new if n_new else 0
  print('\n' + '=' * 70)
  print(f'TOTAL: exact={total["exact"]} padded={total["padded"]} '
        f'skip={total["skip"]}')
  print(f'Transferred {n_transferred}/{n_new} new variables '
        f'({ratio*100:.1f} %).')

  if ratio < 0.8:
    print(_red('FAIL: less than 80 % transferred. Refusing to save.'))
    print('\nSample OLD var names:')
    for v in old_model.weights[:8]:
      print(f'  {v.name}  shape={v.shape}')
    print('Sample NEW var names:')
    for v in new_model.weights[:8]:
      print(f'  {v.name}  shape={v.shape}')
    sys.exit(1)
  print(_green('OK: transfer coverage healthy.'))

  if args.dry_run:
    print('\n' + _yellow('DRY RUN — not saving.'))
    return

  # ── 5. Save ───────────────────────────────────────────────────────────
  print('\n' + '=' * 70)
  print('STEP 5: Saving expanded checkpoint')
  print('=' * 70)
  os.makedirs(args.output_dir, exist_ok=True)

  with strategy.scope():
    fresh_opt = tf.keras.optimizers.Adam(1e-5)
  fresh_opt.build(new_model.trainable_variables)

  new_ckpt = tf.train.Checkpoint(model=new_model, optimizer=fresh_opt)
  manager = tf.train.CheckpointManager(
      new_ckpt, directory=args.output_dir, max_to_keep=5)
  saved_path = manager.save(checkpoint_number=phase1_step)
  print(f'  Saved → {saved_path}')

  # Verify zero-pad rows present.
  print('\n  Verifying zero-pad rows...')
  verify_model = build_and_forward(strategy)
  verify_ckpt = tf.train.Checkpoint(model=verify_model)
  verify_ckpt.restore(saved_path).expect_partial()
  out_stack_kernels = [v for v in verify_model.decoder.out_stack.weights
                       if v.name.endswith('kernel:0')]
  if out_stack_kernels:
    k0 = out_stack_kernels[0].numpy()
    n_zero_rows = int(np.sum(np.all(k0 == 0,
                                    axis=tuple(range(1, k0.ndim)))))
    print(f'  out_stack first kernel shape: {k0.shape}, '
          f'zero rows at axis-0 end: {n_zero_rows}')
    if n_zero_rows >= 512:
      print(_green('  OK: ≥512 zero rows confirmed.'))
    else:
      print(_yellow(f'  warn: expected ≥512 zero rows, found {n_zero_rows}.'))

  print('\n' + '=' * 70)
  print(_green('DONE.'))
  print('=' * 70)
  print(f'\nLaunch Phase 2 with:\n'
        f'  ddsp_run --mode=train \\\n'
        f'    --save_dir={args.output_dir} \\\n'
        f'    --restore_dir={args.output_dir} \\\n'
        f'    --gin_file=papers/basswave/basswave_44k.gin \\\n'
        f'    --gin_file=papers/basswave/basswave_ram_budget.gin \\\n'
        f'    --gin_file={args.phase2_gin} \\\n'
        f'    --gin_param="train.batch_size=2"')


if __name__ == '__main__':
  main()
