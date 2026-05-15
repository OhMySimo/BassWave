# BassWave MIDI Head — Deployment

## Files

```
midi_head.py                              (NEW)   ddsp/training/midi_head.py
denoising_autoencoder.py                  (PATCH) ddsp/training/models/
basswave_midi_head_phase1.gin             (NEW)   gin/papers/basswave/
basswave_midi_head_phase2.gin             (NEW)
smoke_test_midi_head.py                   (NEW)   ddsp/
```

`data_basswave.py` deve già avere `BassWaveWithSidecarProvider` (patched
in fase precedente). `basswave_infer.py` da estendere a fine training
(vedi §7).

## Architettura

```
INPUT audio_clean
       ↓
[DegradationPipeline]
       ↓ audio_corrupted
features['audio'], features['audio_16k']
       ↓
[Preprocessor] → f0_hz, loudness_db (CREPE)
       ↓
[Encoder MfccTimeDistributedRnnEncoder] → z [B, 200, 48]
       ↓
[MIDIHead U-Net 1D]  input: log-mel di audio_16k
   ├→ onset_logit, onset_subframe
   ├→ silence_logit
   ├→ pitch_logits (45 classes)
   ├→ velocity, ks_logits
   └→ midi_cond [B, T, 60]  concat probabilità
       ↓
[Pitch fusion]  CREPE+head → f0_hz aggiornato (Phase 2/inference)
       ↓
[Decoder RnnFcDecoder]  input_keys = ('ld_scaled', 'f0_scaled', 'z', 'midi_cond')
       ↓
[ProcessorGroup]  Harmonic + FilteredNoise + Add
       ↓
audio_synth
       ↓
[Audio-rate silence gate]  ×(1 - sigmoid(silence_logit))
       ↓
FINAL audio (inference)
```

## Procedura

### 0. Verifica sidecar v2

Da `prepare_midi_sidecar_v2.py`:

```bash
ls /media/simone/NVME/MidiDataset/BassWave_TFR_MIDI_v2 | head
```

Numero shard deve combaciare col main TFR (64 train, 4 eval).

### 1. Applica le patch

```bash
cp midi_head.py /path/to/ddsp/training/
cp denoising_autoencoder.py /path/to/ddsp/training/models/
cp basswave_midi_head_phase{1,2}.gin /path/to/ddsp/training/gin/papers/basswave/
cp smoke_test_midi_head.py /path/to/ddsp/
```

Aggiungi in `ddsp/training/__init__.py`:

```python
from ddsp.training import midi_head
```

### 2. Smoke test

```bash
cd /path/to/ddsp
python smoke_test_midi_head.py \
    --main_pattern    '/media/simone/NVME/MidiDataset/BassWave_TFR/basswave-train-*.tfrecord' \
    --sidecar_pattern '/media/simone/NVME/MidiDataset/BassWave_TFR_MIDI_v2/basswave-train-midi-*.tfrecord' \
    --batch_size 4 \
    --plot_out   /tmp/smoke_midi.png
```

4 TEST devono passare. Il plot mostra head predictions UNTRAINED — onset
e pitch saranno random (~0.5 ovunque). Normale.

### 3. Setup runs/basswave_v4_phase1

```bash
V3=/media/simone/NVME/runs/basswave_v3
V4_P1=/media/simone/NVME/runs/basswave_v4_phase1
mkdir -p $V4_P1

CKPT_STEP=75000  # scegli step v3 di partenza
cp $V3/ckpt-${CKPT_STEP}.* $V4_P1/
cat > $V4_P1/checkpoint <<EOF
model_checkpoint_path: "ckpt-${CKPT_STEP}"
all_model_checkpoint_paths: "ckpt-${CKPT_STEP}"
EOF
```

### 4. Phase 1 — Warm-up MIDI head (15-20k step)

```bash
ddsp_run \
  --mode=train \
  --save_dir=/media/simone/NVME/runs/basswave_v4_phase1 \
  --restore_dir=/media/simone/NVME/runs/basswave_v4_phase1 \
  --gin_file=papers/basswave/basswave_44k.gin \
  --gin_file=papers/basswave/basswave_ram_budget.gin \
  --gin_file=papers/basswave/basswave_midi_head_phase1.gin \
  --gin_param="train.batch_size=2"
```

TensorBoard monitor:
- `midi/onset`   < 0.5 a 5k, < 0.3 a 15k
- `midi/silence` < 0.2 a 15k
- `midi/pitch`   < 1.5 (45-class CE, random = ~3.8)
- `total_loss` può rimanere stabile (spectral non si aggiorna, encoder+
  decoder freezati). NORMALE.

Stop quando le `midi/*` losses si stabilizzano.

### 5. Setup runs/basswave_v4_phase2

```bash
V4_P1=/media/simone/NVME/runs/basswave_v4_phase1
V4_P2=/media/simone/NVME/runs/basswave_v4_phase2
mkdir -p $V4_P2

CKPT_STEP_P1=20000  # step finale Phase 1
cp $V4_P1/ckpt-${CKPT_STEP_P1}.* $V4_P2/
cat > $V4_P2/checkpoint <<EOF
model_checkpoint_path: "ckpt-${CKPT_STEP_P1}"
all_model_checkpoint_paths: "ckpt-${CKPT_STEP_P1}"
EOF
```

### 6. Phase 2 — Joint training (80-100k step)

```bash
ddsp_run \
  --mode=train \
  --save_dir=/media/simone/NVME/runs/basswave_v4_phase2 \
  --restore_dir=/media/simone/NVME/runs/basswave_v4_phase2 \
  --gin_file=papers/basswave/basswave_44k.gin \
  --gin_file=papers/basswave/basswave_ram_budget.gin \
  --gin_file=papers/basswave/basswave_midi_head_phase2.gin \
  --gin_param="train.batch_size=2"
```

Monitor:
- `total_loss` scende (spectral attiva su decoder unfreezed)
- `midi/*` può oscillare (distribution shift midi_cond GT → predicted)
- Eval samples: silenzi rispettati, transienti precisi, pitch stabile.

### 7. Inferenza

`basswave_infer.py` da estendere. Quando il modello ha `midi_head`:
sostituire `model(batch, training=False)` con `model.call_with_preset()`
che gestisce silence gate + pitch fusion + midi_cond automaticamente.

Patch minimale a `run_inference()`:

```python
has_head = getattr(model, 'midi_head', None) is not None

if z_preset is None:
    if has_head:
        # Path attraverso call_with_preset: gestisce gate + fusion.
        features = dict(batch)
        outputs = model.call_with_preset(
            features, z_preset=None,
            use_silence_gate=use_silence_gate,
            use_pitch_fusion=True)
        audio_np = outputs['audio_synth'].numpy()
    else:
        outputs, _ = model(batch, return_losses=True, training=False)
        audio_np = model.get_audio_from_outputs(outputs).numpy()
else:
    features = dict(batch)
    z_tiled = tf.tile(z_preset, [B, 1, 1])
    outputs = model.call_with_preset(
        features, z_preset=z_tiled,
        use_silence_gate=use_silence_gate,
        use_pitch_fusion=True)
    audio_np = outputs['audio_synth'].numpy()
```

## Troubleshooting

**OOM in Phase 1**: prova `--gin_param="MIDIHead.bigru_units=128"` (riduce
~1M params), o `--gin_param="MIDIHead.ch_bottleneck=384"` (riduce ~0.5M).

**`midi/pitch` non scende in Phase 1**: verifica smoke test (plot 4 deve
mostrare GT pitch class sensato, note bass tipiche 25-40 in classe).
Se OK, controlla che la head abbia learning rate effettivo non-zero —
nessun layer freezato per errore (`tf.print(midi_head.trainable_variables)`).

**Audio si rompe in Phase 2**: decoder fatica col distribution shift.
- LR più basso: `--gin_param="learning_rate=3e-6"`
- Estendi Phase 1 di altri 5-10k step
- Verifica `grad_clip_norm` non maschera explosion (riduci a 1.0)

**Silence gate troppo aggressivo**: setta
`DenoisingAutoencoder.silence_gate_floor = 0.05` (= -26 dB bleed-through).

**Pitch fusion peggiora alcuni file**: probabilmente file con pitch reale
fuori range 21-64 (es. armonici bassi). Disabilita:
`--gin_param="DenoisingAutoencoder.use_pitch_fusion=False"` in inferenza.
