#!/usr/bin/env bash
# run_phase1.sh — BassWave v4 Phase 1: MIDI head warm-up (backbone frozen).
#
# Prerequisiti:
#   1. Checkpoint v3 (backbone timbric-trained) copiato in PHASE1_DIR:
#        cp -r /media/simone/NVME/runs/basswave_v3/ckpt-XXXXX* $PHASE1_DIR/
#        cp /media/simone/NVME/runs/basswave_v3/checkpoint $PHASE1_DIR/
#   2. Sidecar TFR v2 presenti in SIDECAR_DIR (vedi nota sotto su regenerazione).
#   3. midi_head.py, midi_features.py aggiornati con le patch.
#
# Durata stimata: ~15-20k step (~4-6 ore su RTX 3080).
# Stop quando le loss convergono:
#   midi/onset   < 0.25
#   midi/silence < 0.10
#   midi/pitch   < 1.0
# (monitorare TensorBoard: tensorboard --logdir $PHASE1_DIR)

set -euo pipefail

PHASE1_DIR="/media/simone/NVME/runs/basswave_v4_phase1"
DDSP_ROOT="/app"   # adjust if different

mkdir -p "$PHASE1_DIR"

# Sanity check: checkpoint v3 deve essere presente.
if [ ! -f "$PHASE1_DIR/checkpoint" ]; then
  echo "ERRORE: checkpoint v3 non trovato in $PHASE1_DIR"
  echo "Copiare il checkpoint v3 prima di avviare Phase 1."
  echo "  cp -r /media/simone/NVME/runs/basswave_v3/ckpt-* $PHASE1_DIR/"
  echo "  cp /media/simone/NVME/runs/basswave_v3/checkpoint $PHASE1_DIR/"
  exit 1
fi

echo "=== BassWave v4 — Phase 1: MIDI head warm-up ==="
echo "    save_dir:   $PHASE1_DIR"
echo "    num_steps:  20000"
echo "    batch_size: 2"
echo ""

cd "$DDSP_ROOT"
unset LD_LIBRARY_PATH   # evita conflitti con librerie CUDA di sistema

ddsp_run \
  --mode=train \
  --save_dir="$PHASE1_DIR" \
  --restore_dir="$PHASE1_DIR" \
  --gin_file=ddsp/training/gin/papers/basswave/basswave_44k.gin \
  --gin_file=ddsp/training/gin/papers/basswave/basswave_ram_budget.gin \
  --gin_file=ddsp/training/gin/papers/basswave/phase1_patched.gin \
  --gin_param="train.batch_size=2"

echo ""
echo "=== Phase 1 completata. ==="
echo "Eseguire transition_to_phase2.sh per preparare il checkpoint Phase 2."
