#!/usr/bin/env python3
"""BassWave v3 — inference su WAV con preset z-override opzionale.

Spezza il WAV in chunk da 4 s, calcola f0/loudness on-the-fly,
fa il forward pass con training=False (nessun DegradationPipeline),
e riassembla con cross-fade.

Modalità preset  (--preset <nome>):
  Il vettore z (timbro) non viene estratto dall'input ma calcolato
  come media di N esempi di quel preset presi dai TFRecord.
  f0 e loudness rimangono quelli dell'input — solo il timbro cambia.

Usage:

  # Lista preset disponibili:
    python3 basswave_infer.py --list_presets \\
        --tfrecord_pattern '/media/simone/NVME/MidiDataset/BassWave_TFR/basswave-train-*.tfrecord'

  # Inferenza standard (z dall'input):
    python3 basswave_infer.py \\
        --run_dir /media/simone/NVME/runs/basswave_v3 \\
        --input   /path/to/bass.wav \\
        --output  /path/to/out.wav

  # z forzato al preset "Modern":
    python3 basswave_infer.py \\
        --run_dir /media/simone/NVME/runs/basswave_v3 \\
        --input   /path/to/bass.wav \\
        --output  /path/to/out_modern.wav \\
        --preset  Modern \\
        --tfrecord_pattern '/media/simone/NVME/MidiDataset/BassWave_TFR/basswave-train-*.tfrecord' \\
        --n_preset_samples 64
"""

import argparse
import glob
import json
import os
import re
import sys
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gin
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sig
import tensorflow as tf

DDSP_PATH = '/app'
GIN_PATH  = os.path.join(DDSP_PATH, 'ddsp', 'training', 'gin')
if DDSP_PATH not in sys.path:
    sys.path.insert(0, DDSP_PATH)

from ddsp.training import models, data_basswave, preprocessing
import ddsp.spectral_ops as spec_ops

# ── costanti dal gin config ────────────────────────────────────────────────
SR          = 44100
FRAME_RATE  = 50
HOP         = SR // FRAME_RATE           # 882
CHUNK_AUDIO = 4 * SR + HOP              # 177282
FEAT_LEN    = 4 * FRAME_RATE + 1        # 201
SR_CREPE    = 16000
HOP_CREPE   = SR_CREPE // FRAME_RATE    # 320
CHUNK_16K   = 4 * SR_CREPE + HOP_CREPE  # 64320


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint / model
# ══════════════════════════════════════════════════════════════════════════════

def list_checkpoints(run_dir):
    result = []
    for f in sorted(glob.glob(os.path.join(run_dir, 'ckpt-*.index'))):
        m = re.search(r'ckpt-(\d+)\.index$', f)
        if m:
            result.append((int(m.group(1)), f[:-len('.index')]))
    return sorted(result)


def find_operative_config(run_dir):
    configs = sorted(
        glob.glob(os.path.join(run_dir, 'operative_config-*.gin')),
        key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
    if not configs:
        raise FileNotFoundError(f'Nessun operative_config in {run_dir}')
    return configs[-1]


def load_model(run_dir):
    op_cfg = find_operative_config(run_dir)
    print(f'[config] {os.path.basename(op_cfg)}')
    gin.clear_config()
    gin.add_config_file_search_path(GIN_PATH)
    with gin.unlock_config():
        gin.parse_config_file(op_cfg, skip_unknown=True)
    return models.get_model()


# ══════════════════════════════════════════════════════════════════════════════
# Manifest / preset
# ══════════════════════════════════════════════════════════════════════════════

def load_manifest(tfrecord_pattern):
    """Carica basswave_manifest.json dalla directory dei TFRecord."""
    d = (tfrecord_pattern if os.path.isdir(tfrecord_pattern)
         else os.path.dirname(tfrecord_pattern))
    path = os.path.join(d, 'basswave_manifest.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'basswave_manifest.json non trovato in {d}.\n'
            f'Assicurati che --tfrecord_pattern punti alla directory corretta.')
    with open(path) as f:
        return json.load(f)


def print_presets(manifest):
    p2i    = manifest['preset_to_id']
    counts = manifest.get('preset_counts', {})
    rows   = sorted(p2i.items(), key=lambda kv: -counts.get(kv[0], 0))
    print(f'\n{"Preset":<40} {"ID":>4}  {"Esempi":>7}')
    print('-' * 56)
    for name, pid in rows:
        c = counts.get(name, '?')
        print(f'{name:<40} {pid:>4}  {c:>7}')
    print(f'\nTotale: {len(rows)} preset')


# ══════════════════════════════════════════════════════════════════════════════
# Audio I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_wav_mono_44k(path):
    """Carica WAV → float32 mono @ 44100 Hz."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2_147_483_648.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        print(f'[audio] resample {sr} Hz → {SR} Hz')
        data = sig.resample(data, int(round(len(data) * SR / sr)))
    return data.astype(np.float32)


def save_wav(path, audio):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, SR, (audio * 32767).astype(np.int16))


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def resample_to_16k(chunk_44k):
    return sig.resample(chunk_44k, CHUNK_16K).astype(np.float32)


def _fix_feat(arr, target=FEAT_LEN):
    arr = np.array(arr).squeeze().astype(np.float32)
    if len(arr) > target:
        return arr[:target]
    if len(arr) < target:
        return np.pad(arr, (0, target - len(arr)))
    return arr


def _dither(audio, level=1e-6, seed=None):
    rng = np.random.default_rng(seed)
    return audio + rng.uniform(-level, level, size=audio.shape).astype(np.float32)


def compute_loudness_chunk(chunk_16k):
    """Loudness dal chunk 16 kHz, identico a prepare_basswave.py."""
    seed   = int(np.uint32(chunk_16k[:64].tobytes().__hash__() & 0xFFFFFFFF))
    audio_d = _dither(chunk_16k, level=1e-6, seed=seed)
    ld = spec_ops.compute_loudness(
        audio_d,
        sample_rate=spec_ops.CREPE_SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        padding='center')
    ld = np.array(ld.numpy() if hasattr(ld, 'numpy') else ld).squeeze()
    ld = np.nan_to_num(ld, nan=-120.0, posinf=-120.0, neginf=-120.0)
    return _fix_feat(ld)


def compute_f0_chunk(chunk_16k, viterbi=True):
    """F0 CREPE dal chunk 16 kHz 1-D numpy, identico a prepare_basswave.py."""
    seed    = int(np.uint32(chunk_16k[:64].tobytes().__hash__() & 0xFFFFFFFF))
    audio_d = _dither(chunk_16k, level=1e-6, seed=seed)
    f0, conf = spec_ops.compute_f0(
        audio_d,
        frame_rate=FRAME_RATE,
        viterbi=viterbi,
        padding='center')
    f0   = np.nan_to_num(np.array(f0).squeeze(),   nan=0.0, posinf=0.0)
    conf = np.nan_to_num(np.array(conf).squeeze(),  nan=0.0, posinf=0.0)
    return _fix_feat(f0), _fix_feat(conf)


# ══════════════════════════════════════════════════════════════════════════════
# Chunking / stitching
# ══════════════════════════════════════════════════════════════════════════════

def make_chunks(audio):
    chunks, pos = [], 0
    while pos < len(audio):
        raw   = audio[pos : pos + CHUNK_AUDIO]
        valid = len(raw)
        if valid < CHUNK_AUDIO:
            raw = np.pad(raw, (0, CHUNK_AUDIO - valid))
        chunks.append((raw.astype(np.float32), valid))
        pos += CHUNK_AUDIO
    return chunks


def stitch_crossfade(out_chunks, overlap_samples):
    if not out_chunks:
        return np.array([], dtype=np.float32)
    result = out_chunks[0]
    for nxt in out_chunks[1:]:
        if overlap_samples <= 0 or len(result) < overlap_samples or len(nxt) < overlap_samples:
            result = np.concatenate([result, nxt])
            continue
        fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
        fade_in  = 1.0 - fade_out
        overlap  = result[-overlap_samples:] * fade_out + nxt[:overlap_samples] * fade_in
        result   = np.concatenate([result[:-overlap_samples], overlap, nxt[overlap_samples:]])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Batch builder
# ══════════════════════════════════════════════════════════════════════════════

def build_batch(chunks_data):
    """chunks_data: lista di (c44, c16, f0, conf, ld)."""
    def stack(fn):
        return tf.constant(np.stack([fn(d) for d in chunks_data], axis=0),
                           dtype=tf.float32)
    B = len(chunks_data)
    return {
        'audio':         stack(lambda d: d[0]),
        'audio_16k':     stack(lambda d: d[1]),
        'f0_hz':         stack(lambda d: d[2]),
        'f0_confidence': stack(lambda d: d[3]),
        'loudness_db':   stack(lambda d: d[4]),
        'preset_id':     tf.zeros([B], dtype=tf.int64),
        'transpose':     tf.zeros([B], dtype=tf.int64),
        'groove_cat_id': tf.zeros([B], dtype=tf.int64),
        'file_hash':     tf.zeros([B], dtype=tf.int64),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Preset z computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_preset_z(model, preset_name, manifest, tfrecord_pattern,
                     n_samples=64, mode='time_invariant'):
    """
    Calcola z_preset come media di n_samples esempi del preset target.

    Usa model.encoder() direttamente (senza preprocessor/degradation)
    in modo da ottenere z dal segnale pulito del TFRecord.

    Returns:
        z_preset: tf.Tensor [1, T, z_dim]  (T=200, z_dim=48).
        Con mode='time_invariant' (default) tutti i T frame contengono lo
        stesso vettore z (broadcast). Con mode='time_varying' la dim
        temporale conserva la traiettoria media (legacy).
    """
    p2i = manifest.get('preset_to_id', {})
    if preset_name not in p2i:
        available = sorted(p2i.keys())
        raise ValueError(
            f'Preset "{preset_name}" non trovato nel manifest.\n'
            f'Disponibili: {available}')
    target_id = p2i[preset_name]
    counts    = manifest.get('preset_counts', {})
    n_avail   = counts.get(preset_name, '?')
    print(f'[preset] "{preset_name}"  id={target_id}  esempi disponibili={n_avail}')
    print(f'[preset] raccolta {n_samples} esempi per z_preset...')

    # Stream dal TFRecord, filtro per preset_id, raccogli n_samples batch-size-1
    provider = data_basswave.BassWaveTFRecordProvider(
        file_pattern=tfrecord_pattern,
        balance_presets=False,
        shuffle_buffer_size=256,
        prefetch_size=4,
        interleave_cycle_length=8,
        interleave_block_length=4,
    )
    ds = provider.get_batch(batch_size=1, shuffle=True, repeats=3,
                            drop_remainder=False)

    z_accum = []
    for batch in ds:
        pid = int(batch['preset_id'].numpy().flat[0])
        if pid != target_id:
            continue

        # FIX: usa model.encode(training=False) invece di model.encoder
        # diretto. Il preprocessor risampla f0/ld a time_steps=200; senza
        # questo passo l'encoder vede f0_scaled a 201 frame (CREPE su 4s
        # @ 50 Hz + centerframe) e produce z a 201 — mismatch col decoder
        # che a inferenza si aspetta 200. training=False → degradation
        # pipeline no-op.
        features = dict(batch)
        features = model.encode(features, training=False)
        z = features['z']            # [1, 200, z_dim]
        z_accum.append(z.numpy())

        collected = len(z_accum)
        if collected % 8 == 0:
            print(f'  {collected}/{n_samples}', flush=True)
        if collected >= n_samples:
            break

    if not z_accum:
        raise RuntimeError(
            f'Nessun esempio trovato per preset_id={target_id}. '
            f'Controlla --tfrecord_pattern.')

    z_stack = np.concatenate(z_accum, axis=0)        # [N, T, z_dim]
    n_frames = z_stack.shape[1]

    if mode == 'time_invariant':
        # Media su SAMPLES E TEMPO -> un solo vettore z, broadcast su 200
        # frame. Cattura "carattere timbrico" del preset senza traiettoria
        # fake che nasce dal mediare time-step in clip con onset/decay in
        # posizioni random. Default consigliato — evita wobble di loudness
        # perche' z e' costante e non interferisce con ld_scaled del chunk
        # reale.
        z_flat   = z_stack.reshape(-1, z_stack.shape[-1])    # [N*T, z_dim]
        z_global = z_flat.mean(axis=0)                       # [z_dim]
        z_mean   = np.broadcast_to(
            z_global[None, None, :],
            (1, n_frames, z_stack.shape[-1])).copy()         # [1, T, z_dim]
        norm_dbg = float(np.linalg.norm(z_global))
    elif mode == 'time_varying':
        # Comportamento legacy: media solo su SAMPLES, mantiene dim temporale.
        # Sconsigliato — produce traiettoria z che non corrisponde a nessuna
        # performance reale -> loudness instabile a inferenza.
        z_mean   = z_stack.mean(axis=0, keepdims=True)       # [1, T, z_dim]
        norm_dbg = float(np.linalg.norm(z_mean.mean(axis=1)))
    else:
        raise ValueError(
            f"mode must be 'time_invariant' or 'time_varying', got {mode!r}")

    z_tensor = tf.constant(z_mean, dtype=tf.float32)
    print(f'[preset] z_preset mode={mode}  '
          f'shape={tuple(z_mean.shape)}  '
          f'norm={norm_dbg:.3f}  (da {len(z_accum)} esempi)')
    return z_tensor


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def _decode_features(model, features, head_outputs=None,
                     use_silence_gate=True):
    """
    Esegue decoder + processor_group su un features dict gi\u00e0 encodato.
    Se head_outputs (dict da MIDIHead/TransientHead) contiene 'silence_logit'
    e use_silence_gate, applica gating audio-rate a audio_synth.

    IMPORTANTE: se head_outputs contiene 'midi_cond' (MIDIHead), lo inietta
    in features prima del decoder. Senza, decoder crash su KeyError perch\u00e9
    Phase 2 decoder ha input_keys=('ld_scaled','f0_scaled','z','midi_cond').

    Restituisce audio_synth [B, n_samples].
    """
    # Iniezione midi_cond per Phase 2 decoder (input_keys con midi_cond).
    if head_outputs is not None and 'midi_cond' in head_outputs:
        features['midi_cond'] = head_outputs['midi_cond']

    features.update(model.decoder(features, training=False))
    pg_out = model.processor_group(features, return_outputs_dict=True)
    audio_synth = pg_out['signal']

    # Audio-rate silence gate.
    if (use_silence_gate
        and head_outputs is not None
        and 'silence_logit' in head_outputs
        and hasattr(model, 'apply_audio_rate_silence_gate')):
        audio_synth = model.apply_audio_rate_silence_gate(
            audio_synth, head_outputs['silence_logit'])

    return audio_synth


def _maybe_run_head(model, features):
    """Esegue head (MIDI o Transient) su features se il modello ne ha una.

    PRIORIT\u00c0: midi_head (Phase 2+) prima di transient_head (Phase 1).
    La head deve girare PRIMA dell'eventuale override di z con z_preset,
    cos\u00ec usa la z ENCODER (timing-aware) e non z_preset (timbre).

    Per MIDIHead: input \u00e8 log_mel da audio_16k, NON dalla z dell'encoder.

    Returns:
      head_outputs: dict con i tensor di output, oppure None.
    """
    # MIDIHead (Phase 2+): legge audio_16k → log_mel.
    midi_head = getattr(model, 'midi_head', None)
    mel_extractor = getattr(model, 'mel_extractor', None)
    if midi_head is not None and mel_extractor is not None:
        if 'audio_16k' not in features:
            return None
        log_mel = mel_extractor(features['audio_16k'])
        # Align time steps if mel framing differs from f0 framing.
        if 'f0_hz' in features:
            T_target = int(features['f0_hz'].shape[1])
            cur_T = int(log_mel.shape[1])
            if cur_T != T_target:
                if cur_T > T_target:
                    log_mel = log_mel[:, :T_target, :]
                else:
                    pad_amt = T_target - cur_T
                    log_mel = tf.pad(log_mel, [[0,0],[0,pad_amt],[0,0]])
        return midi_head(log_mel, training=False)

    # Fallback TransientHead (Phase 1 transient_head, legacy).
    transient_head = getattr(model, 'transient_head', None)
    if transient_head is not None:
        z = features.get('z', None)
        if z is None:
            return None
        return transient_head(z, training=False)

    return None


def _maybe_fuse_pitch_inference(model, features, head_outputs,
                                use_pitch_fusion):
    """Applica pitch fusion CREPE+head a inferenza.

    Modifica features['f0_hz'] e features['f0_scaled'] in-place se la fusion
    \u00e8 abilitata, il modello ha midi_head con pitch_logits, e il flag
    runtime \u00e8 True.
    """
    if not use_pitch_fusion or head_outputs is None:
        return features
    if 'pitch_logits' not in head_outputs or 'silence_logit' not in head_outputs:
        return features
    if 'f0_hz' not in features or 'f0_confidence' not in features:
        return features

    try:
        from ddsp.training.midi_head import fuse_pitch_with_crepe
        from ddsp.training import preprocessing
    except ImportError:
        return features

    f0_fused = fuse_pitch_with_crepe(
        f0_crepe=features['f0_hz'],
        f0_crepe_conf=features['f0_confidence'],
        pitch_logits_head=head_outputs['pitch_logits'],
        silence_logit_head=head_outputs['silence_logit'],
        onset_logit_head=head_outputs.get('onset_logit', None))
    features['f0_hz'] = f0_fused
    features['f0_scaled'] = preprocessing.scale_f0_hz(f0_fused)
    return features


def run_inference(model, ckpt_prefix, all_chunk_data,
                  batch_size=1, overlap_samples=0, z_preset=None,
                  use_silence_gate=True, use_pitch_fusion=True):
    """
    all_chunk_data: list of (c44, c16, f0, conf, ld, valid_len)
    z_preset: tf.Tensor [1, T, z_dim] oppure None (usa z dall'input)
    use_silence_gate: se True e modello ha head, applica silence gate
    use_pitch_fusion: se True e modello ha midi_head con pitch_logits,
                      fonde CREPE+head per correggere octave errors.
    """
    model.restore(ckpt_prefix)
    has_midi_head = getattr(model, 'midi_head', None) is not None
    has_any_head = (has_midi_head
                    or getattr(model, 'transient_head', None) is not None)
    has_pitch = (has_midi_head
                 and getattr(model, 'mel_extractor', None) is not None)
    mode = (f'z_preset="{z_preset is not None}" '
            f'silence_gate="{use_silence_gate and has_any_head}" '
            f'pitch_fusion="{use_pitch_fusion and has_pitch}"')
    print(f'[ckpt] restored  →  {os.path.basename(ckpt_prefix)}  ({mode})')

    out_chunks = []
    n = len(all_chunk_data)

    for start in range(0, n, batch_size):
        batch_items = all_chunk_data[start : start + batch_size]
        B = len(batch_items)
        batch = build_batch([(d[0], d[1], d[2], d[3], d[4])
                             for d in batch_items])

        if z_preset is None:
            # Percorso standard: z estratto dall'input.
            # Costruiamo features esplicitamente per controllare pitch
            # fusion + midi_cond, invece di affidarci a model.call() che
            # ha la pitch fusion baked-in se use_pitch_fusion=True nel gin.
            features = dict(batch)
            features = model.encode(features, training=False)
            head_outputs = _maybe_run_head(model, features)
            features = _maybe_fuse_pitch_inference(
                model, features, head_outputs, use_pitch_fusion)
            audio_synth = _decode_features(
                model, features,
                head_outputs=head_outputs,
                use_silence_gate=use_silence_gate)
            audio_np = audio_synth.numpy()
        else:
            # z-override + gate + fusion manuali.
            features = dict(batch)
            features = model.encode(features, training=False)
            head_outputs = _maybe_run_head(model, features)
            features = _maybe_fuse_pitch_inference(
                model, features, head_outputs, use_pitch_fusion)

            # Override z per decoder (timbre da preset).
            z_tiled = tf.tile(z_preset, [B, 1, 1])
            features['z'] = z_tiled

            audio_synth = _decode_features(
                model, features,
                head_outputs=head_outputs,
                use_silence_gate=use_silence_gate)
            audio_np = audio_synth.numpy()

        for j, item in enumerate(batch_items):
            valid = item[5]
            out_chunks.append(audio_np[j, :valid])

        end_idx = min(start + batch_size, n)
        print(f'  chunks {start+1}–{end_idx}/{n}', flush=True)

    return stitch_crossfade(out_chunks, overlap_samples)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir',            default=None)
    parser.add_argument('--input',              default=None)
    parser.add_argument('--output',             default=None)
    parser.add_argument('--ckpt_step',          default='latest')
    parser.add_argument('--overlap_ms',         type=float, default=20.0)
    parser.add_argument('--batch_size',         type=int,   default=1)
    parser.add_argument('--no_viterbi',         action='store_true')
    # Preset z-override
    parser.add_argument('--preset',             default=None,
                        help='Nome preset per z-override (es. "Modern")')
    parser.add_argument('--tfrecord_pattern',   default=None,
                        help='Glob TFRecord (obbligatorio con --preset / --list_presets)')
    parser.add_argument('--n_preset_samples',   type=int, default=64,
                        help='Esempi da mediare per z_preset (default 64)')
    parser.add_argument('--z_preset_mode',
                        choices=['time_invariant', 'time_varying'],
                        default='time_invariant',
                        help=('time_invariant (default): media z anche sul '
                              'tempo, un vettore broadcast su tutti i frame. '
                              'time_varying: media solo sui sample, mantiene '
                              'dim temporale (legacy, causa wobble di '
                              'loudness perche z evolve in modo scorrelato '
                              'dal chunk reale).'))
    parser.add_argument('--list_presets',       action='store_true',
                        help='Stampa preset disponibili e termina')
    parser.add_argument('--no_silence_gate',    action='store_true',
                        help='Disabilita silence gate audio-rate '
                             '(default: gate attivo se modello ha TransientHead). '
                             'Utile per A/B confronto pre/post head.')
    parser.add_argument('--no_pitch_fusion',    action='store_true',
                        help='Disabilita pitch fusion CREPE+head a inferenza. '
                             'Utile per A/B + debug octave errors. La fusion '
                             'a inferenza puo\' essere pi\u00f9 aggressiva del training '
                             'perch\u00e9 la head \u00e8 piu\' sicura e fa ottava sopra/sotto.')
    args = parser.parse_args()

    viterbi = not args.no_viterbi

    # ── --list_presets ─────────────────────────────────────────────────────
    if args.list_presets:
        if not args.tfrecord_pattern:
            parser.error('--list_presets richiede --tfrecord_pattern')
        manifest = load_manifest(args.tfrecord_pattern)
        print_presets(manifest)
        return

    # ── validazione argomenti obbligatori ──────────────────────────────────
    for flag, val in [('--run_dir', args.run_dir),
                      ('--input',   args.input),
                      ('--output',  args.output)]:
        if val is None:
            parser.error(f'{flag} è obbligatorio')
    if args.preset and not args.tfrecord_pattern:
        parser.error('--preset richiede --tfrecord_pattern')

    # ── checkpoint ─────────────────────────────────────────────────────────
    checkpoints = list_checkpoints(args.run_dir)
    if not checkpoints:
        raise RuntimeError(f'Nessun checkpoint in {args.run_dir}')
    if args.ckpt_step == 'latest':
        step, prefix = checkpoints[-1]
    else:
        step = int(args.ckpt_step)
        matches = [(s, p) for s, p in checkpoints if s == step]
        if not matches:
            raise ValueError(f'Step {step} non trovato. '
                             f'Disponibili: {[s for s,_ in checkpoints]}')
        step, prefix = matches[0]
    print(f'[ckpt] step={step}  →  {prefix}')

    # ── modello ────────────────────────────────────────────────────────────
    model = load_model(args.run_dir)

    # ── preset z (opzionale) ───────────────────────────────────────────────
    z_preset = None
    if args.preset:
        model.restore(prefix)   # necessario prima di chiamare encoder
        manifest = load_manifest(args.tfrecord_pattern)
        z_preset = compute_preset_z(
            model, args.preset, manifest,
            args.tfrecord_pattern, args.n_preset_samples,
            mode=args.z_preset_mode)

    # ── audio ──────────────────────────────────────────────────────────────
    print(f'\n[audio] loading  {args.input}')
    audio   = load_wav_mono_44k(args.input)
    n_total = len(audio)
    print(f'[audio] {n_total/SR:.2f}s  ({n_total} campioni @ {SR} Hz)')

    raw_chunks = make_chunks(audio)
    print(f'[chunks] {len(raw_chunks)} × {CHUNK_AUDIO/SR:.2f}s')

    # ── feature extraction ─────────────────────────────────────────────────
    print(f'\n[features] f0 (CREPE viterbi={viterbi}) + loudness...')
    all_chunk_data = []
    for idx, (c44, valid) in enumerate(raw_chunks):
        c16      = resample_to_16k(c44)
        ld       = compute_loudness_chunk(c16)
        f0, conf = compute_f0_chunk(c16, viterbi=viterbi)
        conf_mask = conf > 0.5
        if conf_mask.any():
            info = f'f0={f0[conf_mask].mean():.1f} Hz  conf={conf_mask.mean()*100:.0f}%'
        else:
            info = 'conf bassa (silenzio?)'
        print(f'  chunk {idx+1:3d}/{len(raw_chunks)}  {info}', flush=True)
        all_chunk_data.append((c44, c16, f0, conf, ld, valid))

    # ── inference ──────────────────────────────────────────────────────────
    overlap_samples = int(args.overlap_ms * SR / 1000)
    preset_label    = f'preset="{args.preset}"' if args.preset else 'z=input'
    print(f'\n[infer] {preset_label}  batch={args.batch_size}  '
          f'overlap={args.overlap_ms} ms...')

    result = run_inference(
        model, prefix, all_chunk_data,
        batch_size=args.batch_size,
        overlap_samples=overlap_samples,
        z_preset=z_preset,
        use_silence_gate=not args.no_silence_gate,
        use_pitch_fusion=not args.no_pitch_fusion)

    result = result[:n_total]

    # ── salva ──────────────────────────────────────────────────────────────
    save_wav(args.output, result)
    peak_db = 20 * np.log10(np.abs(result).max() + 1e-9)
    print(f'\n[done]  {len(result)/SR:.2f}s  peak={peak_db:.1f} dBFS')
    print(f'        → {args.output}')


if __name__ == '__main__':
    main()
