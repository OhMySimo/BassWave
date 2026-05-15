"""Scan BassWave TFRecord shards and report the distribution of f0_hz lengths.

Usage:
    python scan_tfrecords.py \
        --tfrecord_dir=/media/simone/NVME/MidiDataset/BassWave_TFR \
        --split=train          # or eval
        --max_per_shard=50     # records to inspect per shard (0 = all)
        --workers=8            # parallel shard workers
"""

import argparse
import glob
import os
import sys
import concurrent.futures
from collections import Counter, defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

# ── expected constants (must match gin / data_basswave.py) ──────────────────
EXPECTED_AUDIO_LEN   = 177282   # 4s * 44100 + 882  (centered)
EXPECTED_AUDIO16K_LEN = 64320   # 4s * 16000 + 320  (centered)
EXPECTED_FEAT_LEN    = 201      # get_framed_lengths(64320, 1024, 320, 'center')

ALL_KEYS = [
    'audio', 'audio_16k',
    'f0_hz', 'f0_confidence', 'loudness_db',
    'preset_id', 'transpose', 'groove_cat_id', 'file_hash',
]

FLOAT_KEYS   = {'audio', 'audio_16k', 'f0_hz', 'f0_confidence', 'loudness_db'}
INT_KEYS     = {'preset_id', 'transpose', 'groove_cat_id', 'file_hash'}


def inspect_record(raw: bytes):
    """Return a dict of issues for one raw serialised Example, or None if clean."""
    ex = tf.train.Example()
    try:
        ex.ParseFromString(raw)
    except Exception as e:
        return {'parse_error': str(e)}

    issues = {}
    feats = ex.features.feature

    for key in ALL_KEYS:
        if key not in feats:
            issues[key] = 'MISSING'
            continue

        feat = feats[key]
        if key in FLOAT_KEYS:
            vals = feat.float_list.value
            length = len(vals)
            if key == 'audio' and length != EXPECTED_AUDIO_LEN:
                issues[key] = f'len={length} (expected {EXPECTED_AUDIO_LEN})'
            elif key == 'audio_16k' and length != EXPECTED_AUDIO16K_LEN:
                issues[key] = f'len={length} (expected {EXPECTED_AUDIO16K_LEN})'
            elif key in ('f0_hz', 'f0_confidence', 'loudness_db') and length != EXPECTED_FEAT_LEN:
                issues[key] = f'len={length} (expected {EXPECTED_FEAT_LEN})'
        else:  # int scalar
            vals = feat.int64_list.value
            if len(vals) == 0:
                issues[key] = 'EMPTY_INT'

    return issues if issues else None


def scan_shard(path: str, max_records: int):
    """Scan one shard. Returns (shard_path, total_records, bad_records_list)."""
    bad = []
    total = 0
    try:
        for raw in tf.data.TFRecordDataset(path):
            raw_bytes = raw.numpy()
            total += 1
            issues = inspect_record(raw_bytes)
            if issues:
                bad.append((total, issues))
            if max_records > 0 and total >= max_records:
                break
    except Exception as e:
        bad.append((total, {'shard_error': str(e)}))
    return path, total, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfrecord_dir',
                    default='/media/simone/NVME/MidiDataset/BassWave_TFR')
    ap.add_argument('--split', default='train', choices=['train', 'eval', 'both'])
    ap.add_argument('--max_per_shard', type=int, default=100,
                    help='Max records to inspect per shard (0=all, slow)')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    patterns = []
    if args.split in ('train', 'both'):
        patterns.append(os.path.join(args.tfrecord_dir, 'basswave-train-*.tfrecord'))
    if args.split in ('eval', 'both'):
        patterns.append(os.path.join(args.tfrecord_dir, 'basswave-eval-*.tfrecord'))

    shards = []
    for pat in patterns:
        shards.extend(sorted(glob.glob(pat)))

    if not shards:
        print(f'ERROR: no TFRecords found in {args.tfrecord_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(shards)} shards  (max_per_shard={args.max_per_shard or "ALL"})')
    print()

    total_records  = 0
    total_bad      = 0
    key_issue_ctr  = Counter()          # key → count of bad records
    len_dist       = defaultdict(Counter) # key → Counter of lengths seen

    # Collect f0_hz length distribution across ALL records (good and bad)
    # for a quick histogram. Use a small separate pass on first shard only.
    print('Checking f0_hz length distribution on first shard …')
    first_shard = shards[0]
    f0_lens = Counter()
    audio_lens = Counter()
    for raw in tf.data.TFRecordDataset(first_shard):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        f0_len = len(ex.features.feature.get('f0_hz', tf.train.Feature()).float_list.value)
        a_len  = len(ex.features.feature.get('audio', tf.train.Feature()).float_list.value)
        f0_lens[f0_len] += 1
        audio_lens[a_len] += 1
    print(f'  Shard 0 f0_hz lengths : {dict(f0_lens)}')
    print(f'  Shard 0 audio  lengths: {dict(audio_lens)}')
    print()

    print(f'Full scan with {args.workers} workers …')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(scan_shard, s, args.max_per_shard): s for s in shards}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            shard_path, n, bad = fut.result()
            total_records += n
            total_bad     += len(bad)
            done += 1
            for _, issues in bad:
                for key in issues:
                    key_issue_ctr[key] += 1
            if done % 10 == 0 or done == len(shards):
                print(f'  [{done}/{len(shards)}] records so far: {total_records}, '
                      f'bad: {total_bad}', flush=True)

    print()
    print('=' * 60)
    print(f'TOTAL records inspected : {total_records}')
    print(f'TOTAL bad records found : {total_bad}')
    if key_issue_ctr:
        print()
        print('Issues by key:')
        for key, cnt in key_issue_ctr.most_common():
            pct = 100.0 * cnt / max(total_records, 1)
            print(f'  {key:<20s}  {cnt:>6d} bad  ({pct:.1f}%)')
    else:
        print()
        print('No issues found — dataset looks clean.')
    print('=' * 60)


if __name__ == '__main__':
    main()
