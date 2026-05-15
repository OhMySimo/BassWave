#!/usr/bin/env bash
# run_phase3.sh — Launch BassWave v4 Phase 3 self-distillation.
#
# Phase 3 = z-aux self-distillation. Parte da P1 (124k step). Niente
# chirurgia decoder. Encoder + decoder unfreeze per la prima volta dopo v3.

set -euo pipefail

PHASE1_DIR="/media/simone/NVME/runs/basswave_v4_phase1"
PHASE3_DIR="/media/simone/NVME/runs/basswave_v4_phase3"
SIDECAR_DIR="/media/simone/NVME/MidiDataset/BassWave_TFR_MIDI_v2"
MAIN_TFR_DIR="/media/simone/NVME/MidiDataset/BassWave_TFR"

BASE_GIN="papers/basswave/basswave_44k.gin"
RAM_BUDGET_GIN="papers/basswave/basswave_ram_budget.gin"
PHASE3_GIN="papers/basswave/basswave_midi_head_phase3_distill.gin"

BATCH_SIZE="${PHASE3_BATCH_SIZE:-2}"
DDSP_GIN_ROOT="/app/ddsp/training/gin"

RED=$'\033[91m'; GREEN=$'\033[92m'; YELLOW=$'\033[93m'; CYAN=$'\033[96m'; NC=$'\033[0m'
info()  { echo "${CYAN}[INFO]${NC}  $*"; }
warn()  { echo "${YELLOW}[WARN]${NC}  $*"; }
ok()    { echo "${GREEN}[OK]${NC}    $*"; }
fail()  { echo "${RED}[FAIL]${NC}  $*" >&2; exit 1; }

# ─── Pre-flight ───────────────────────────────────────────────────────────
info "Pre-flight checks per Phase 3..."

# 1. Phase 1 dir has a ckpt.
[[ -d "$PHASE1_DIR" ]] || fail "PHASE1_DIR non esiste: $PHASE1_DIR"
P1_CKPT_INDEX=$(ls -1t "$PHASE1_DIR"/ckpt-*.index 2>/dev/null | head -1 || true)
[[ -n "$P1_CKPT_INDEX" ]] || fail "Nessun ckpt in $PHASE1_DIR."
P1_CKPT_STEP=$(basename "$P1_CKPT_INDEX" .index | sed 's/^ckpt-//')
ok "Phase 1 ckpt: ckpt-$P1_CKPT_STEP"

# 2. Setup PHASE3_DIR.
if [[ ! -d "$PHASE3_DIR" ]]; then
  info "PHASE3_DIR non esiste. Lo creo e copio ckpt-$P1_CKPT_STEP da P1."
  mkdir -p "$PHASE3_DIR"
  cp "$PHASE1_DIR/ckpt-$P1_CKPT_STEP".* "$PHASE3_DIR/"
  cat > "$PHASE3_DIR/checkpoint" <<EOF
model_checkpoint_path: "ckpt-$P1_CKPT_STEP"
all_model_checkpoint_paths: "ckpt-$P1_CKPT_STEP"
EOF
  ok "Phase 3 dir inizializzato in $PHASE3_DIR"
else
  P3_CKPT_INDEX=$(ls -1t "$PHASE3_DIR"/ckpt-*.index 2>/dev/null | head -1 || true)
  if [[ -z "$P3_CKPT_INDEX" ]]; then
    warn "PHASE3_DIR esiste ma vuoto. Copio ckpt-$P1_CKPT_STEP da P1."
    cp "$PHASE1_DIR/ckpt-$P1_CKPT_STEP".* "$PHASE3_DIR/"
    cat > "$PHASE3_DIR/checkpoint" <<EOF
model_checkpoint_path: "ckpt-$P1_CKPT_STEP"
all_model_checkpoint_paths: "ckpt-$P1_CKPT_STEP"
EOF
  else
    P3_CKPT_STEP=$(basename "$P3_CKPT_INDEX" .index | sed 's/^ckpt-//')
    ok "PHASE3_DIR esistente con ckpt-$P3_CKPT_STEP (resume)"
  fi
fi

# 3. Phase 3 gin file exists.
if [[ -d "$DDSP_GIN_ROOT" ]]; then
  GIN_PATH="$DDSP_GIN_ROOT/$PHASE3_GIN"
  [[ -f "$GIN_PATH" ]] || fail "Gin file mancante: $GIN_PATH"
  ok "Trovato $GIN_PATH"
  grep -q "z_aux_projector" "$GIN_PATH" || \
    fail "$PHASE3_GIN non contiene z_aux_projector. Wrong file?"
  grep -q "RnnFcDecoder.input_keys" "$GIN_PATH" || \
    fail "$PHASE3_GIN non setta RnnFcDecoder.input_keys"
  if grep -q "midi_cond" <(grep "RnnFcDecoder.input_keys" "$GIN_PATH"); then
    fail "$PHASE3_GIN ha 'midi_cond' nei decoder input_keys! Wrong gin — Phase 3 NON deve avere midi_cond al decoder."
  fi
  ok "input_keys decoder = 3 (P1 compatible, no surgery)"
fi

# 4. Sidecar TFR.
N_SIDECAR=$(ls -1 "$SIDECAR_DIR"/basswave-train-midi-*.tfrecord 2>/dev/null | wc -l)
N_MAIN=$(ls -1 "$MAIN_TFR_DIR"/basswave-train-*.tfrecord 2>/dev/null | wc -l)
if [[ "$N_SIDECAR" -eq 0 ]] || [[ "$N_SIDECAR" -ne "$N_MAIN" ]]; then
  fail "Sidecar TFR mismatch: $N_SIDECAR sidecar vs $N_MAIN main."
fi
ok "Sidecar TFR: $N_SIDECAR shard"

# 5. ddsp_run.
command -v ddsp_run &>/dev/null || fail "ddsp_run non nel PATH."
ok "ddsp_run disponibile"

# 6. Verifica che zaux_projector.py sia importabile.
if [[ -d "/app/ddsp" ]]; then
  PYTHONPATH=/app python3 -c "from ddsp.training import zaux_projector" 2>/dev/null \
    && ok "zaux_projector module importabile" \
    || warn "zaux_projector.py NON trovato in /app/ddsp/training/. Copia il file prima di lanciare."
fi

# ─── Launch ───────────────────────────────────────────────────────────────
echo
info "Configurazione Phase 3:"
echo "    PHASE3_DIR      = $PHASE3_DIR"
echo "    Batch size      = $BATCH_SIZE"
echo "    Strategia       = z-aux self-distillation"
echo "    Decoder         = unfreezed, input_keys=3 (P1 compatible)"
echo "    Encoder         = unfreezed"
echo "    Projector       = NEW (random-init)"
echo "    Gin stack       = $BASE_GIN + $RAM_BUDGET_GIN + $PHASE3_GIN"
echo

info "Comando:"
cat <<EOF

  ddsp_run \\
    --mode=train \\
    --save_dir=$PHASE3_DIR \\
    --restore_dir=$PHASE3_DIR \\
    --gin_file=$BASE_GIN \\
    --gin_file=$RAM_BUDGET_GIN \\
    --gin_file=$PHASE3_GIN \\
    --gin_param="train.batch_size=$BATCH_SIZE"

EOF

read -r -p "Avvio Phase 3 training? [Y/n] " CONFIRM
CONFIRM=${CONFIRM:-Y}
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  info "Aborted."
  exit 0
fi

echo
info "TB summaries in $PHASE3_DIR/summaries/"
echo
info "Metriche critiche da monitorare:"
echo "  spectral_loss  — deve SCENDERE (decoder finalmente unfreezed)"
echo "                   target: < 5.80 entro 20k step, < 5.50 entro 100k"
echo "  z_aux/mse      — parte alto (projector random), poi scende"
echo "                   target: < 0.5 entro 20k step"
echo "  midi/*         — restano stabili o leggermente in calo (head trained)"
echo
info "Segnali di problema (stop e diagnostica):"
echo "  • spectral_loss SALE > 6.5 → LR troppo alto, riduci a 5e-6"
echo "  • z_aux/mse plateau alto (>2.0) → projector non converge, aumenta hidden_dim"
echo "  • spectral scende ma z_aux/mse va a 0 → z collapse, abbassa lambda_aux"
echo "  • midi/* salgono molto → head viene corrotta, abbassa midi loss weights"
echo
sleep 2

exec ddsp_run \
  --mode=train \
  --save_dir="$PHASE3_DIR" \
  --restore_dir="$PHASE3_DIR" \
  --gin_file="$BASE_GIN" \
  --gin_file="$RAM_BUDGET_GIN" \
  --gin_file="$PHASE3_GIN" \
  --gin_param="train.batch_size=$BATCH_SIZE"
