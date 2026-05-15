#!/usr/bin/env bash
# run_phase2.sh — Launch BassWave v4 Phase 2 joint fine-tune.
#
# Prerequisiti:
#   1. Phase 1 training completato in $PHASE1_DIR
#   2. Surgery del decoder gia' eseguita:
#      python expand_decoder_for_midi_cond.py \
#          --phase1_dir $PHASE1_DIR \
#          --output_dir $PHASE2_DIR \
#          --phase1_gin papers/basswave/phase1_patched.gin \
#          --phase2_gin papers/basswave/basswave_midi_head_phase2.gin
#   3. Sidecar TFR v2 in $SIDECAR_DIR
#
# Pre-flight: lo script verifica che il ckpt surgeriato esista e che
# papers/basswave/basswave_midi_head_phase2.gin sia nel gin tree.

set -euo pipefail

# ─── Config (modifica solo questi paths se la tua struttura è diversa) ────
PHASE1_DIR="/media/simone/NVME/runs/basswave_v4_phase1"
PHASE2_DIR="/media/simone/NVME/runs/basswave_v4_phase2"
SIDECAR_DIR="/media/simone/NVME/MidiDataset/BassWave_TFR_MIDI_v2"
MAIN_TFR_DIR="/media/simone/NVME/MidiDataset/BassWave_TFR"

# Gin files (paths relativi al gin search path di ddsp_run).
BASE_GIN="papers/basswave/basswave_44k.gin"
RAM_BUDGET_GIN="papers/basswave/basswave_ram_budget.gin"
PHASE2_GIN="papers/basswave/basswave_midi_head_phase2.gin"

# Override gin params (modificabili a riga di comando: PHASE2_BATCH_SIZE=4 ./run_phase2.sh).
BATCH_SIZE="${PHASE2_BATCH_SIZE:-2}"

# Path al gin tree dentro al modulo ddsp (per pre-flight check).
DDSP_GIN_ROOT="/app/ddsp/training/gin"

# ─── Color helpers ────────────────────────────────────────────────────────
RED=$'\033[91m'; GREEN=$'\033[92m'; YELLOW=$'\033[93m'; CYAN=$'\033[96m'; NC=$'\033[0m'

info()  { echo "${CYAN}[INFO]${NC}  $*"; }
warn()  { echo "${YELLOW}[WARN]${NC}  $*"; }
ok()    { echo "${GREEN}[OK]${NC}    $*"; }
fail()  { echo "${RED}[FAIL]${NC}  $*" >&2; exit 1; }

# ─── Pre-flight checks ────────────────────────────────────────────────────
info "Pre-flight checks..."

# 1. Phase 2 dir esiste e contiene il ckpt surgeriato.
if [[ ! -d "$PHASE2_DIR" ]]; then
  fail "PHASE2_DIR non esiste: $PHASE2_DIR
       Esegui prima la surgery:
           python expand_decoder_for_midi_cond.py \\
               --phase1_dir $PHASE1_DIR \\
               --output_dir $PHASE2_DIR \\
               --phase1_gin papers/basswave/phase1_patched.gin \\
               --phase2_gin $PHASE2_GIN"
fi

CKPT_INDEX=$(ls -1 "$PHASE2_DIR"/ckpt-*.index 2>/dev/null | head -1 || true)
if [[ -z "$CKPT_INDEX" ]]; then
  fail "Nessun ckpt-*.index in $PHASE2_DIR. Surgery non eseguita?"
fi

CKPT_STEP=$(basename "$CKPT_INDEX" .index | sed 's/^ckpt-//')
ok "Trovato ckpt surgeriato: ckpt-$CKPT_STEP"

# 2. File `checkpoint` index esiste e punta al ckpt corretto.
if [[ ! -f "$PHASE2_DIR/checkpoint" ]]; then
  warn "File 'checkpoint' index mancante. Lo creo io ora puntando a ckpt-$CKPT_STEP."
  cat > "$PHASE2_DIR/checkpoint" <<EOF
model_checkpoint_path: "ckpt-$CKPT_STEP"
all_model_checkpoint_paths: "ckpt-$CKPT_STEP"
EOF
  ok "Scritto $PHASE2_DIR/checkpoint"
else
  POINTED=$(grep -oP 'model_checkpoint_path:\s*"\K[^"]+' "$PHASE2_DIR/checkpoint" || true)
  if [[ "$POINTED" != "ckpt-$CKPT_STEP" ]]; then
    warn "checkpoint index punta a '$POINTED' ma trovato 'ckpt-$CKPT_STEP'. Aggiorno."
    cat > "$PHASE2_DIR/checkpoint" <<EOF
model_checkpoint_path: "ckpt-$CKPT_STEP"
all_model_checkpoint_paths: "ckpt-$CKPT_STEP"
EOF
  fi
fi

# 3. Gin file di Phase 2 esiste.
if [[ -d "$DDSP_GIN_ROOT" ]]; then
  GIN_PATH="$DDSP_GIN_ROOT/$PHASE2_GIN"
  if [[ ! -f "$GIN_PATH" ]]; then
    fail "Gin file Phase 2 mancante: $GIN_PATH"
  fi
  ok "Trovato $GIN_PATH"

  # Quick sanity: il gin deve specificare input_keys con 4 elementi.
  if ! grep -q "RnnFcDecoder.input_keys.*midi_cond" "$GIN_PATH"; then
    fail "$PHASE2_GIN non contiene 'RnnFcDecoder.input_keys = ...midi_cond...'.
         Stai usando il file giusto?"
  fi
  ok "RnnFcDecoder.input_keys include 'midi_cond' in $PHASE2_GIN"
else
  warn "DDSP_GIN_ROOT non trovato a $DDSP_GIN_ROOT. Skip gin file check."
fi

# 4. Sidecar TFR esistono.
N_SIDECAR=$(ls -1 "$SIDECAR_DIR"/basswave-train-midi-*.tfrecord 2>/dev/null | wc -l)
N_MAIN=$(ls -1 "$MAIN_TFR_DIR"/basswave-train-*.tfrecord 2>/dev/null | wc -l)
if [[ "$N_SIDECAR" -eq 0 ]]; then
  fail "Nessun sidecar TFR in $SIDECAR_DIR. Esegui prima prepare_midi_sidecar_v2.py."
fi
if [[ "$N_SIDECAR" -ne "$N_MAIN" ]]; then
  fail "Sidecar shard count ($N_SIDECAR) != main shard count ($N_MAIN).
       Re-esegui prepare_midi_sidecar_v2.py con --num_shards $N_MAIN."
fi
ok "Sidecar TFR: $N_SIDECAR shard train (allineati al main)"

# 5. ddsp_run nel PATH.
if ! command -v ddsp_run &>/dev/null; then
  fail "ddsp_run non nel PATH. Sei dentro al container Docker?"
fi
ok "ddsp_run disponibile: $(command -v ddsp_run)"

# ─── Launch ───────────────────────────────────────────────────────────────
echo
info "Configurazione finale:"
echo "    PHASE2_DIR      = $PHASE2_DIR"
echo "    Restoring from  = ckpt-$CKPT_STEP"
echo "    Batch size      = $BATCH_SIZE"
echo "    Gin stack       = $BASE_GIN + $RAM_BUDGET_GIN + $PHASE2_GIN"
echo

# Stampa il comando esatto che verra' eseguito.
info "Comando:"
cat <<EOF

  ddsp_run \\
    --mode=train \\
    --save_dir=$PHASE2_DIR \\
    --restore_dir=$PHASE2_DIR \\
    --gin_file=$BASE_GIN \\
    --gin_file=$RAM_BUDGET_GIN \\
    --gin_file=$PHASE2_GIN \\
    --gin_param="train.batch_size=$BATCH_SIZE"

EOF

read -r -p "Avvio training? [Y/n] " CONFIRM
CONFIRM=${CONFIRM:-Y}
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  info "Aborted."
  exit 0
fi

echo
info "Lancio Phase 2 training. TB summaries in $PHASE2_DIR/summaries/"
info "Per monitorare in un altro terminale:"
echo "    tensorboard --logdir=$PHASE2_DIR/summaries --port=6006"
echo
info "Check critici nei primi 500 step di TB:"
echo "    spectral_loss @ step $CKPT_STEP : deve essere ~6.0 (= fine Phase 1)"
echo "    midi/silence  @ step $CKPT_STEP : deve essere ~0.15"
echo "    Se uno dei due e' molto piu' alto -> surgery rotta, ferma e diagnostica."
echo
sleep 2

exec ddsp_run \
  --mode=train \
  --save_dir="$PHASE2_DIR" \
  --restore_dir="$PHASE2_DIR" \
  --gin_file="$BASE_GIN" \
  --gin_file="$RAM_BUDGET_GIN" \
  --gin_file="$PHASE2_GIN" \
  --gin_param="train.batch_size=$BATCH_SIZE"
