#!/usr/bin/env python3
"""Esegui test_midi_alignment su N file casuali dal dataset.

Usage:
    python run_alignment_sample.py \
        --metadata_jsonl /media/simone/NVME/MidiDataset/metadata.jsonl \
        --output_dir     ./alignment_checks \
        --n              10 \
        --override_bpm   120 \
        --offset_ms      6.35 \
        --keymap         /media/simone/NVME/MidiDataset/bass_midi_keymap.json
"""
import argparse, json, os, random, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument('--metadata_jsonl', required=True)
ap.add_argument('--output_dir', default='./alignment_checks')
ap.add_argument('--n', type=int, default=10)
ap.add_argument('--override_bpm', type=float, default=120.0)
ap.add_argument('--offset_ms', type=float, default=0.0,
                help='Offset costante audio-MIDI in ms (da measure_midi_offset).')
ap.add_argument('--keymap', default='',
                help='Path bass_midi_keymap.json. Se fornito, abilita '
                     'separazione KS/note e Plot 4 in test_midi_alignment.')
ap.add_argument('--src_prefix', default='E:/')
ap.add_argument('--dst_prefix', default='/media/simone/NVME/')
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

def tr(p):
    return (args.dst_prefix + p[len(args.src_prefix):]).replace('\\', '/') \
        if p.startswith(args.src_prefix) else p

rows = []
with open(args.metadata_jsonl) as f:
    rows = [json.loads(l) for l in f]

random.seed(args.seed)
sample = random.sample(rows, min(args.n, len(rows)))
os.makedirs(args.output_dir, exist_ok=True)

script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'test_midi_alignment.py')

ks_label = ('with keymap' if args.keymap else 'NO keymap (KS=notes)')
print(f'Running alignment on {len(sample)} files '
      f'(bpm={args.override_bpm}, offset={args.offset_ms:+.2f}ms, {ks_label})...\n')

for i, row in enumerate(sample):
    flac = tr(row['flac_path'])
    midi = tr(row['midi_path'])
    if not os.path.exists(flac):
        print(f'[{i+1}] SKIP (missing): {os.path.basename(flac)}')
        continue
    if not os.path.exists(midi):
        print(f'[{i+1}] SKIP (no midi): {os.path.basename(midi)}')
        continue

    label = f'{i+1:02d}_{os.path.basename(flac).replace(".flac","")}'[:60]
    out_png = os.path.join(args.output_dir, f'{label}.png')

    cmd = [
        sys.executable, script,
        '--midi',         midi,
        '--flac',         flac,
        '--output',       out_png,
        '--override_bpm', str(args.override_bpm),
        '--offset_ms',    str(args.offset_ms),
    ]
    if args.keymap:
        cmd += ['--keymap', args.keymap]

    print(f'[{i+1}/{len(sample)}] {os.path.basename(flac)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if any(x in line for x in ['note events', 'KS events', 'windows=',
                                    'silence', 'onsets', 'ok]']):
            print(f'  {line.strip()}')
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr.splitlines()[-1] if result.stderr else "?"}')
    else:
        print(f'  -> {out_png}')

print(f'\nDone. PNG files in: {args.output_dir}')
