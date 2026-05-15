#!/usr/bin/env python3
"""Offline validation script for BassWave DDSP checkpoints.

Per ogni checkpoint trovato in uno o più run_dirs:
  1. Carica il modello dal operative_config del run
  2. Fa inference su N batch del validation set
  3. Calcola spectral loss (stesso setup del training)
  4. Salva audio WAV (original / degraded / synthesized) per il primo batch
  5. Produce un grafico matplotlib della loss vs step

Usage (dentro il container Docker):
    python3 basswave_validate.py \
        --run_dirs /media/simone/NVME/runs/basswave_v1 \
                   /media/simone/NVME/runs/basswave_v2 \
        --out_dir  /media/simone/NVME/runs/validation_results \
        --num_batches 20 \
        --batch_size 2

Output:
    <out_dir>/
        basswave_v1/
            ckpt-22000/
                original_0.wav
                degraded_0.wav   (se DenoisingAutoencoder)
                synthesized_0.wav
            ckpt-24000/
                ...
        basswave_v2/
            ...
        eval_loss_curve.png
"""

import argparse
import glob

import os
import re
import sys
import time
import logging

# Sopprimi i warning "Value in checkpoint could not be found" per le
# variabili dell'optimizer (non servono per inference, sono attesi).
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf

# ---------------------------------------------------------------------------
# Percorsi del codice DDSP (aggiusta se necessario)
# ---------------------------------------------------------------------------
DDSP_PATH = '/app'
GIN_PATH  = os.path.join(DDSP_PATH, 'ddsp', 'training', 'gin')

if DDSP_PATH not in sys.path:
    sys.path.insert(0, DDSP_PATH)

from ddsp.training import data_basswave, models, train_util
import ddsp


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def list_checkpoints(run_dir):
    """Restituisce lista di (step, ckpt_prefix) ordinata per step."""
    index_files = sorted(
        glob.glob(os.path.join(run_dir, 'ckpt-*.index')))
    result = []
    for f in index_files:
        m = re.search(r'ckpt-(\d+)\.index$', f)
        if m:
            step = int(m.group(1))
            prefix = f[:-len('.index')]
            result.append((step, prefix))
    return sorted(result, key=lambda x: x[0])


def find_operative_config(run_dir):
    """Trova l'operative_config più recente nel run_dir."""
    configs = sorted(
        glob.glob(os.path.join(run_dir, 'operative_config-*.gin')),
        key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
    if not configs:
        raise FileNotFoundError(
            f'Nessun operative_config in {run_dir}')
    return configs[-1]   # il più recente


def load_model_from_config(operative_config_path):
    """Carica il modello DDSP dal operative_config gin."""
    gin.clear_config()
    gin.add_config_file_search_path(GIN_PATH)
    with gin.unlock_config():
        gin.parse_config_file(operative_config_path, skip_unknown=True)
    model = models.get_model()
    return model


def build_eval_dataset(operative_config_path, batch_size):
    """Costruisce il dataset di validazione dal gin config."""
    # I parametri eval_data/* sono già nel operative_config
    # Li leggiamo direttamente dal gin già parsato.
    try:
        provider = gin.get_bindings('eval_data/BassWaveTFRecordProvider')
        file_pattern = provider.get(
            'file_pattern',
            '/media/simone/NVME/MidiDataset/BassWave_TFR/basswave-eval-*.tfrecord')
    except Exception:
        file_pattern = (
            '/media/simone/NVME/MidiDataset/BassWave_TFR/'
            'basswave-eval-*.tfrecord')

    # Costruisce provider con parametri RAM-safe (non usa balanced sampling
    # in eval — inutile e costoso)
    CYCLE = 4
    provider_obj = data_basswave.BassWaveTFRecordProvider(
        file_pattern=file_pattern,
        balance_presets=False,
        shuffle_buffer_size=0,
        prefetch_size=2,
        interleave_cycle_length=CYCLE,
        interleave_block_length=2,
        num_parallel_calls=CYCLE,   # deve essere <= cycle_length
    )
    ds = provider_obj.get_batch(
        batch_size=batch_size,
        shuffle=False,
        repeats=1,
        drop_remainder=False)
    return ds


def audio_to_wav(tensor, sample_rate=44100):
    """Converte tf.Tensor float32 [-1,1] in int16 numpy per scipy."""
    arr = np.array(tensor)
    if arr.ndim == 2:
        arr = arr[0]   # prendi il primo esempio del batch
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767).astype(np.int16), sample_rate


def save_wav(path, tensor, sample_rate=44100):
    data, sr = audio_to_wav(tensor, sample_rate)
    wav.write(path, sr, data)


def compute_spectral_loss(audio_target, audio_gen):
    """SpectralLoss con stesso setup del training."""
    loss_fn = ddsp.losses.SpectralLoss(
        fft_sizes=[8192, 4096, 2048, 1024, 512, 256],
        loss_type='L1',
        mag_weight=1.0,
        logmag_weight=1.0)
    return float(loss_fn(audio_target, audio_gen).numpy())


# ---------------------------------------------------------------------------
# Core: valida un singolo checkpoint
# ---------------------------------------------------------------------------

def validate_checkpoint(ckpt_prefix, run_dir, out_ckpt_dir,
                        model, eval_ds, sample_rate, num_batches,
                        save_audio_batches=1):
    """
    Carica il checkpoint, fa inference su num_batches, restituisce mean loss.
    Salva WAV per i primi save_audio_batches batch.
    """
    os.makedirs(out_ckpt_dir, exist_ok=True)

    # model.restore() è un metodo custom DDSP che non restituisce
    # il TF status object — non supporta .expect_partial().
    # I warning optimizer sono già soppressi da TF_CPP_MIN_LOG_LEVEL=3.
    model.restore(ckpt_prefix)

    losses = []
    for batch_idx, batch in enumerate(eval_ds):
        if batch_idx >= num_batches:
            break

        # Forward pass (no training=True per evitare dropout/augmentation)
        outputs, batch_losses = model(batch, return_losses=True, training=False)
        audio_gen = model.get_audio_from_outputs(outputs)

        # Loss sul clean target
        audio_target = batch['audio']
        loss = compute_spectral_loss(audio_target, audio_gen)
        losses.append(loss)

        # Salva WAV per i primi batch
        if batch_idx < save_audio_batches:
            prefix = os.path.join(out_ckpt_dir, f'batch{batch_idx:02d}')
            save_wav(f'{prefix}_original.wav',   audio_target, sample_rate)
            save_wav(f'{prefix}_synthesized.wav', audio_gen,   sample_rate)
            # Degraded (solo DenoisingAutoencoder)
            if 'audio_corrupted' in outputs:
                save_wav(f'{prefix}_degraded.wav',
                         outputs['audio_corrupted'], sample_rate)

        print(f'    batch {batch_idx+1}/{num_batches}  loss={loss:.4f}')

    mean_loss = float(np.mean(losses)) if losses else float('nan')
    return mean_loss


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_loss_curve(results, out_path):
    """
    results: dict  run_name -> [(step, mean_loss), ...]
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10.colors

    for idx, (run_name, points) in enumerate(sorted(results.items())):
        if not points:
            continue
        points = sorted(points, key=lambda x: x[0])
        steps  = [p[0] for p in points]
        losses = [p[1] for p in points]
        color  = colors[idx % len(colors)]

        ax.plot(steps, losses, 'o-', color=color, linewidth=2,
                markersize=5, label=run_name)

        # Annotazione del minimo
        min_idx = int(np.argmin(losses))
        ax.annotate(
            f'{losses[min_idx]:.4f}',
            xy=(steps[min_idx], losses[min_idx]),
            xytext=(8, -14), textcoords='offset points',
            fontsize=7.5, color=color)

    ax.set_title('Validation Spectral Loss per Checkpoint',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Mean Spectral Loss (eval set)')
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nGrafico salvato: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--run_dirs', nargs='+', required=True,
        help='Uno o più run_dir (es. .../basswave_v1 .../basswave_v2)')
    parser.add_argument(
        '--out_dir', required=True,
        help='Directory di output per WAV e grafico')
    parser.add_argument(
        '--num_batches', type=int, default=20,
        help='Batch di validazione per checkpoint (default 20)')
    parser.add_argument(
        '--batch_size', type=int, default=2,
        help='Batch size (default 2 — safe per VRAM)')
    parser.add_argument(
        '--save_audio_batches', type=int, default=1,
        help='Quanti batch salvare come WAV (default 1)')
    parser.add_argument(
        '--sample_rate', type=int, default=44100)
    parser.add_argument(
        '--skip_existing', action='store_true', default=True,
        help='Salta checkpoint già validati (default True)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_file = os.path.join(args.out_dir, 'val_losses.txt')

    # Carica risultati precedenti se esistono
    all_results = {}   # run_name -> [(step, loss)]
    if os.path.exists(results_file) and args.skip_existing:
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) == 3:
                    run_name, step, loss = parts
                    all_results.setdefault(run_name, []).append(
                        (int(step), float(loss)))
        print(f'Caricati {sum(len(v) for v in all_results.values())} '
              f'risultati precedenti da {results_file}')

    with open(results_file, 'a') as log:
        for run_dir in args.run_dirs:
            run_dir  = run_dir.rstrip('/')
            run_name = os.path.basename(run_dir)
            checkpoints = list_checkpoints(run_dir)

            if not checkpoints:
                print(f'[warn] Nessun checkpoint in {run_dir}')
                continue

            print(f'\n=== Run: {run_name} ({len(checkpoints)} checkpoints) ===')

            # Carica operative_config una sola volta per run
            try:
                op_config = find_operative_config(run_dir)
                print(f'Operative config: {os.path.basename(op_config)}')
            except FileNotFoundError as e:
                print(f'[error] {e}')
                continue

            model = load_model_from_config(op_config)
            eval_ds = build_eval_dataset(op_config, args.batch_size)
            already_done = {s for s, _ in all_results.get(run_name, [])}

            for step, ckpt_prefix in checkpoints:
                if step in already_done and args.skip_existing:
                    print(f'  ckpt-{step}: già validato, skip')
                    continue

                ckpt_out = os.path.join(args.out_dir, run_name, f'ckpt-{step}')
                print(f'  ckpt-{step} ...', flush=True)
                t0 = time.time()

                try:
                    mean_loss = validate_checkpoint(
                        ckpt_prefix, run_dir, ckpt_out,
                        model, eval_ds,
                        args.sample_rate, args.num_batches,
                        args.save_audio_batches)
                    elapsed = time.time() - t0
                    print(f'  ckpt-{step}: mean_loss={mean_loss:.4f}  '
                          f'({elapsed:.0f}s)')

                    all_results.setdefault(run_name, []).append(
                        (step, mean_loss))
                    log.write(f'{run_name}\t{step}\t{mean_loss:.6f}\n')
                    log.flush()

                except Exception as exc:
                    print(f'  [error] ckpt-{step}: {exc}')
                    import traceback; traceback.print_exc()

    # Plot finale
    plot_path = os.path.join(args.out_dir, 'eval_loss_curve.png')
    plot_loss_curve(all_results, plot_path)

    # Riepilogo testuale
    print('\n=== Riepilogo ===')
    for run_name, points in sorted(all_results.items()):
        if not points:
            continue
        points = sorted(points)
        best_step, best_loss = min(points, key=lambda x: x[1])
        last_step, last_loss = points[-1]
        print(f'{run_name}: best={best_loss:.4f} @ ckpt-{best_step}  '
              f'last={last_loss:.4f} @ ckpt-{last_step}')


if __name__ == '__main__':
    main()
