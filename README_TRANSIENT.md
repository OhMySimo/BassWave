# BassWave TransientHead — Deployment Operations

End-to-end procedura per integrare la `TransientHead` nel training v4.

## Files consegnati

```
midi_features.py                  (gia' tuo)
prepare_midi_sidecar.py           (gia' tuo)
test_midi_alignment.py            (gia' tuo)
run_alignment_sample.py           (gia' tuo)
bass_midi_keymap.json             (gia' tuo)

transient_head.py                 (NEW)   ddsp/training/transient_head.py
denoising_autoencoder.py          (PATCH) ddsp/training/models/denoising_autoencoder.py
data_basswave_sidecar_patch.py    (PATCH) da appendere in fondo a data_basswave.py
basswave_infer.py                 (PATCH) ddsp/basswave_infer.py
basswave_transient_head.gin       (NEW)   gin/papers/basswave/basswave_transient_head.gin
smoke_test_transient_head.py      (NEW)   ddsp/smoke_test_transient_head.py
```

## Procedura

### 1. Genera sidecar TFRecord (~30 min)

```bash
python prepare_midi_sidecar.py \
    --input_root      /media/simone/NVME/MidiDataset/FLAC_AUG \
    --midi_root       /media/simone/NVME/MidiDataset/MIDI_AUG \
    --metadata_jsonl  /media/simone/NVME/MidiDataset/metadata.jsonl \
    --manifest_json   /media/simone/NVME/MidiDataset/BassWave_TFR/basswave_manifest.json \
    --keymap_json     /media/simone/NVME/MidiDataset/bass_midi_keymap.json \
    --output_dir      /media/simone/NVME/MidiDataset/BassWave_TFR_MIDI \
    --num_shards      64 \
    --eval_split_fraction 0.05 \
    --override_bpm    120 \
    --offset_const_ms 6.0
```

Verifica esito:

```bash
ls /media/simone/NVME/MidiDataset/BassWave_TFR_MIDI/ | head
# basswave-train-midi-00000-of-00064.tfrecord ...
# basswave-eval-midi-00000-of-00004.tfrecord ...
```

Numero shard deve combaciare col main TFR (64 train, 4 eval).

### 2. Applica le patch

#### transient_head.py

```bash
cp transient_head.py /path/to/ddsp/training/
```

Aggiungi import in `ddsp/training/__init__.py`:

```python
from ddsp.training import transient_head
```

#### denoising_autoencoder.py

```bash
cp denoising_autoencoder.py \
   /path/to/ddsp/training/models/denoising_autoencoder.py
```

(Sostituisce il file esistente — backup il precedente se vuoi.)

#### data_basswave.py

Apri `data_basswave_sidecar_patch.py` e appendi la classe `BassWaveWithSidecarProvider` in fondo a `ddsp/training/data_basswave.py`.

#### basswave_infer.py

```bash
cp basswave_infer.py /path/to/ddsp/
```

(Sostituisce il file esistente.)

#### gin overlay

```bash
cp basswave_transient_head.gin \
   /path/to/ddsp/training/gin/papers/basswave/
```

### 3. Smoke test (~1 min)

PRIMA di lanciare 100k step di training, verifica che join + head + loss
non abbiano bug:

```bash
cd /path/to/ddsp
python smoke_test_transient_head.py \
    --main_pattern    '/media/simone/NVME/MidiDataset/BassWave_TFR/basswave-train-*.tfrecord' \
    --sidecar_pattern '/media/simone/NVME/MidiDataset/BassWave_TFR_MIDI/basswave-train-midi-*.tfrecord' \
    --batch_size 4 \
    --plot_out   /tmp/smoke_align.png
```

Output atteso: 4 TEST con `OK`, plus un PNG di verifica visiva.

Se fallisce: il messaggio di errore dice esattamente cosa controllare.

### 4. Prepara dir training v4

Restore weights da v3 (no optimizer, no head). La TransientHead avra' pesi
nuovi inizializzati a caso:

```bash
mkdir -p /media/simone/NVME/runs/basswave_v4
cp /media/simone/NVME/runs/basswave_v3/ckpt-<STEP>.* /media/simone/NVME/runs/basswave_v4/
cat > /media/simone/NVME/runs/basswave_v4/checkpoint <<EOF
model_checkpoint_path: "ckpt-<STEP>"
all_model_checkpoint_paths: "ckpt-<STEP>"
EOF
```

### 5. Lancia training v4

```bash
ddsp_run \
  --mode=train \
  --save_dir=/media/simone/NVME/runs/basswave_v4 \
  --restore_dir=/media/simone/NVME/runs/basswave_v4 \
  --gin_file=papers/basswave/basswave_44k.gin \
  --gin_file=papers/basswave/basswave_ram_budget.gin \
  --gin_file=papers/basswave/basswave_v3_strong_ramp.gin \
  --gin_file=papers/basswave/basswave_transient_head.gin \
  --gin_param="train.batch_size=2" \
  --gin_param="DegradationPipeline.min_n_apply=4" \
  --gin_param="DegradationPipeline.max_n_apply=6"
```

Monitora TensorBoard per i nuovi scalari:
- `transient/onset`
- `transient/silence`
- `transient/subframe_on`
- ...

I valori utili di riferimento (dopo ~10k step):
- `transient/silence` < 0.2 = decente
- `transient/onset` < 0.5 (focal-weighted BCE)
- `transient/subframe_on` < 0.05

### 6. Inferenza con silence gate

```bash
# Standard (gate attivo se modello ha head)
python basswave_infer.py \
    --run_dir /media/simone/NVME/runs/basswave_v4 \
    --input   ./infer_test/input.wav \
    --output  ./infer_test/out_with_gate.wav \
    --preset  Modern \
    --tfrecord_pattern '/media/simone/NVME/MidiDataset/BassWave_TFR/basswave-train-*.tfrecord' \
    --n_preset_samples 64

# A/B: stesso modello senza gate per confronto
python basswave_infer.py \
    --run_dir /media/simone/NVME/runs/basswave_v4 \
    --input   ./infer_test/input.wav \
    --output  ./infer_test/out_no_gate.wav \
    --preset  Modern \
    --tfrecord_pattern '...' \
    --no_silence_gate
```

Confronta i due output: con gate, le pause devono essere effettivamente
silenziose; senza gate, dovresti sentire il "MIDI piatto" che avevi prima.

## Notes

- Il sidecar provider e' compatibile con runs senza head: setta semplicemente
  `BassWaveWithSidecarProvider.sidecar_file_pattern = None` nel gin e legge
  solo i TFR principali.
- Per validation con metriche head (precision/recall onset), serve estendere
  `basswave_validate.py` — non incluso in questa consegna, da fare in seguito
  se utile.
- Per disabilitare la testa a runtime (es. ablation A/B sul modello stesso),
  setta `DenoisingAutoencoder.use_silence_gate = False` nel gin.

## Architettura riassunto

```
INPUT AUDIO
    ↓
[Preprocessor]
    ↓ (audio_corrupted via DegradationPipeline)
[Encoder]
    ↓ z [B, 200, 48]
   ├─→ [TransientHead]
   │       ↓ silence_logit, onset_logit, ...
   │       └→ (training)  TransientLoss
   │       └→ (inference) audio-rate silence gate
   │
   └→ [Decoder]
          ↓ amps, harmonic_distribution, noise_magnitudes
        [ProcessorGroup]
          ↓ audio_synth [B, 177282]
          ✕ silence_gate (1 - sigmoid(silence_logit)) upsampled to audio rate
          ↓
        FINAL OUTPUT
```
