# BassWave DDSP Patch

Drop-in extension to Magenta DDSP for restoring/resynthesising 44.1 kHz mono
bass guitar tracks. Implements the Luke-Ditria-style denoising training pattern
(corrupted input → clean target) on top of DDSP's structured
harmonic + filtered-noise synthesis.

## What this changes

| File | New / Modified | Purpose |
|------|---------------|---------|
| `ddsp/training/data_basswave.py` | new | TFRecord provider for BassWave layout, with optional preset-balanced sampling |
| `ddsp/training/preprocessing_denoise.py` | new | `BassDegradation` (lowpass / quantize / noise / clip / dropout) + `DenoisingF0LoudnessPreprocessor` (clean → corrupted online) |
| `ddsp/training/models/denoising_autoencoder.py` | new | Autoencoder subclass whose loss target is `audio_clean` not `audio` |
| `ddsp/training/data_preparation/prepare_basswave.py` | new | Walks `FLAC_AUG/<groove>/<pattern>/<bpm>/Variation_NN__Tp±N__Pk<preset>.flac`, runs CREPE + loudness, writes TFRecord shards with metadata |
| `ddsp/training/gin/datasets/basswave.gin` | new | Dataset config @ 44.1 kHz, `frame_rate=50`, `centered=True` |
| `ddsp/training/gin/models/basswave_denoise.gin` | new | Model config: `z_dim=48`, `n_harmonics=256`, `noise_bands=256`, multi-scale spectral loss with FFT sizes scaled for 44.1 k |
| `ddsp/training/gin/papers/basswave/basswave_44k.gin` | new | End-to-end pipeline: dataset + model + optimisation (`LR=3e-5`, `batch_size=8`, `grad_clip=2.0`) |
| `ddsp/training/__init__.py` | edit | Register `data_basswave`, `preprocessing_denoise` |
| `ddsp/training/models/__init__.py` | edit | Register `DenoisingAutoencoder` |

## Apply the patch

```bash
git clone --depth 1 https://github.com/magenta/ddsp.git
cd ddsp

# Copy the new + modified files in:
cp -rv /path/to/ddsp_basswave_patch/* .

# Sanity-check overlay (no other files should differ):
git status
git diff ddsp/training/__init__.py ddsp/training/models/__init__.py

# Install in editable mode:
pip install -e .
pip install crepe pydub apache-beam
```

## Step 1 — Prepare TFRecords

```bash
python -m ddsp.training.data_preparation.prepare_basswave \
  --input_root=/media/simone/NVME/MidiDataset/FLAC_AUG \
  --output_dir=/media/simone/NVME/MidiDataset/BassWave_TFR \
  --num_shards=64 \
  --eval_split_fraction=0.05 \
  --crepe_model=full \
  --viterbi=True
```

Output: `basswave-train-NNNNN-of-NNNNN.tfrecord` and
`basswave-eval-NNNNN-of-NNNNN.tfrecord` in `--output_dir`.

Estimated time: ~30 min on a desktop with CREPE-full on GPU; ~3 h on CPU.

**Important — low-F0 caveat:** CREPE's training distribution is centred well
above the BassWave median F0 of 65 Hz. Spot-check a few samples after
prep to confirm `f0_hz` median tracks the actual fundamental and isn't pegged
to the 2nd harmonic. If it is, options are:
1. Limit f0 to a tighter range via post-hoc clipping in the prep script
   (a `f0_max_hz` flag — easy to add).
2. Use `pyworld` / `pyin` instead of CREPE for f0 extraction (more accurate
   for bass).
3. Fine-tune CREPE on bass — out of scope for this patch.

## Step 2 — Train

```bash
ddsp_run \
  --mode=train \
  --save_dir=/media/simone/NVME/runs/basswave_v1 \
  --gin_file=papers/basswave/basswave_44k.gin \
  --gin_param="batch_size=8" \
  --gin_param="train.num_steps=500000"
```

Memory footprint at `batch_size=8`, `n_samples=177282`, `z_time_steps=250`:
roughly 18-22 GB on a 24 GB GPU (RTX 3090/4090). Reduce to `batch_size=4` if
OOM, and double `lr_decay_steps` to compensate.

## Step 3 — Evaluate

```bash
ddsp_run \
  --mode=eval \
  --save_dir=/media/simone/NVME/runs/basswave_v1 \
  --gin_file=papers/basswave/basswave_44k.gin
```

## Step 4 — Inference / restoration

At inference time the `DenoisingF0LoudnessPreprocessor.degradation` is
**not** invoked (it's gated on `training=True`). The pipeline becomes:

```
real degraded audio → encoder(audio) → z
+ external f0_hz, loudness_db (you compute these)
→ decoder → ProcessorGroup → restored audio
```

You need to provide `f0_hz` and `loudness_db` for the input. Two options:

1. **Run CREPE on the degraded input** — quality depends on how degraded.
   For mild noise/lowpass this works.
2. **Provide an oracle f0** (e.g. from a synced MIDI file or an external
   tracker). The model will resynthesise the bass following that pitch
   contour with the timbre captured by `z`.

A worked inference notebook would mirror `ddsp/colab/demos/timbre_transfer.ipynb`
with the dataset path swapped in. The `Autoencoder.encode/decode` methods are
the same — `DenoisingAutoencoder` only overrides loss-target selection.

## Hyperparameter rationale

All the non-default values trace back to the BassWave analysis:

| Param | Value | Why |
|-------|-------|-----|
| `z_dims` | 48 | dim@95% PCA = 40, +8 for temporal MFCC variance, rounded up |
| `n_harmonics` | 256 | At F0=65 Hz, 256 harmonics → 16.6 kHz, 75% of Nyquist |
| `noise_bands` | 256 | HNR=1.3 dB, noise channel carries equal energy to harmonics |
| `frame_rate` | 50 | Matches `vst_48k.gin`; 20 ms hop, fine for 65 Hz F0 |
| `frame_size` | 2048 | 46 ms, ≥3 cycles at 65 Hz (1024 was 23 ms = 1.5 cycles, marginal) |
| `fft_sizes` | `[8192,4096,2048,1024,512,256]` | Scaled ~5x from 16 k DDSP defaults |
| `use_angular_cumsum` | True | Required at SR > 16 kHz (oscillator phase precision) |
| `LR` | 3e-5 | DDSP paper value; `3e-4` is unstable with the corrupt→clean objective |
| `grad_clip_norm` | 2.0 | F0 spikes from CREPE on low-F0 bass cause occasional explosions |
| `clean_prob` | 0.15 | Some clean passthroughs prevent the network from "always denoising" |

## Known limitations / things to watch

1. **F0 estimation is the bottleneck.** If CREPE confuses fundamental with
   2nd harmonic, the synth will resonate at the wrong frequency and the
   spectral loss will explode. Monitor `f0_confidence` in summaries.
2. **Preset imbalance.** `Modern=2148` vs `multi_fx1=1`. The
   `balance_presets=True` flag in the train provider rescales sampling by
   `1/sqrt(count)`, but a preset with 1 sample will still be undersampled
   relative to its representativeness target. Consider dropping presets
   with <50 samples for the first run.
3. **No paired clean/degraded data.** All training degradation is
   synthetic. If your real-world degradation differs substantially from
   `BassDegradation`'s primitives (e.g. you want to restore vintage tape
   compression artefacts), edit that layer to match.
4. **The `BassDegradation.call_with_clean_mix` is the real entry point.**
   The `call` method on the layer is for stand-alone use; the preprocessor
   uses `call_with_clean_mix` to preserve `audio_clean`.
5. **No reverb in the default DAG.** If your bass recordings have room,
   add `@effects.FilteredNoiseReverb()` to the `ProcessorGroup.dag` like
   `vst_48k.gin` does.

## What this is NOT

This is a structured DDSP-style **resynthesiser**. The output `audio_synth`
is generated from scratch by the harmonic + filtered-noise synth — it is
not the input filtered or processed. It preserves f0 and loudness contour
of the input, and reconstructs the timbre that the encoder's `z` describes.

If your goal is true paired-data audio restoration (output stays
sample-accurate to the input but cleaner), DDSP is the wrong tool — you'd
want a waveform-domain U-Net or a neural codec like Encodec.
