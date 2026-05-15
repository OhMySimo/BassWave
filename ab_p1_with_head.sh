#!/usr/bin/env bash
# ab_p1_with_head.sh — A/B test Phase 1 vs Phase 3 (self-distillation).
#
# Genera 4 file per confronto diretto:
#   <basename>_p1_hybrid.wav        P1 puro (decoder v3, timbro baseline)
#   <basename>_p1_preset.wav        P1 preset
#   <basename>_p3_hybrid.wav        P3 (decoder migliorato da distillation)
#   <basename>_p3_preset.wav        P3 preset
#
# Tutti con --no_silence_gate --no_pitch_fusion (post-processing off):
# l'obiettivo è valutare il TIMBRO puro del decoder, non gli effetti della head.
# Se vuoi anche il confronto con head attiva, setta INCLUDE_HEAD_VARIANTS=1.

set -euo pipefail

PHASE1_DIR="${PHASE1_DIR:-/media/simone/NVME/runs/basswave_v4_phase1}"
PHASE3_DIR="${PHASE3_DIR:-/media/simone/NVME/runs/basswave_v4_phase3}"
TFRECORD_PATTERN="${TFRECORD_PATTERN:-/media/simone/NVME/MidiDataset/BassWave_TFR/basswave-train-*.tfrecord}"
PRESET="${PRESET:-Modern}"
N_PRESET_SAMPLES="${N_PRESET_SAMPLES:-64}"
INCLUDE_HEAD_VARIANTS="${INCLUDE_HEAD_VARIANTS:-0}"

INPUT="${1:-}"
OUTPUT_DIR="${2:-./ab_p1_vs_p3}"

if [[ -z "$INPUT" ]]; then
  cat <<EOF
Usage: $0 <input.wav> [output_dir]

Variabili d'ambiente:
  PHASE1_DIR             (default: .../basswave_v4_phase1)
  PHASE3_DIR             (default: .../basswave_v4_phase3)
  PRESET                 (default: Modern)
  N_PRESET_SAMPLES       (default: 64)
  INCLUDE_HEAD_VARIANTS  (default: 0 — setta 1 per aggiungere varianti con head attiva)

Esempi:
  $0 /tmp/bass.wav ./ab_p1_vs_p3
  PRESET=Funk_Fusion INCLUDE_HEAD_VARIANTS=1 $0 input.wav ./ab_funk
EOF
  exit 1
fi

[[ -f "$INPUT" ]] || { echo "ERROR: input not found: $INPUT" >&2; exit 1; }

BASENAME=$(basename "$INPUT" | sed 's/\.[^.]*$//')
mkdir -p "$OUTPUT_DIR"

RED=$'\033[91m'; GREEN=$'\033[92m'; CYAN=$'\033[96m'; YELLOW=$'\033[93m'; NC=$'\033[0m'
info() { echo "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo "${GREEN}[OK]${NC}    $*"; }
warn() { echo "${YELLOW}[WARN]${NC}  $*"; }

run_infer() {
  local run_dir="$1" out_path="$2" extra_args="$3" label="$4"
  info "Rendering: $label → $(basename "$out_path")"
  if python basswave_infer.py \
        --run_dir "$run_dir" --input "$INPUT" --output "$out_path" \
        --tfrecord_pattern "$TFRECORD_PATTERN" \
        --n_preset_samples "$N_PRESET_SAMPLES" \
        $extra_args 2>&1 | tail -4; then
    [[ -f "$out_path" ]] && ok "$out_path" || warn "MISSING: $out_path"
  else
    warn "FAILED: $label"
  fi
  echo
}

# ─── P1 (baseline) ───────────────────────────────────────────────────────
info "═══ Phase 1 — decoder baseline (v3 frozen) ═══"
run_infer "$PHASE1_DIR" "$OUTPUT_DIR/${BASENAME}_p1_hybrid.wav" \
  "--no_silence_gate --no_pitch_fusion" \
  "P1 hybrid (timbro puro)"

run_infer "$PHASE1_DIR" "$OUTPUT_DIR/${BASENAME}_p1_preset.wav" \
  "--preset $PRESET --no_silence_gate --no_pitch_fusion" \
  "P1 preset $PRESET (timbro puro)"

# ─── P3 (self-distillation) ───────────────────────────────────────────────
info "═══ Phase 3 — decoder migliorato (self-distillation) ═══"
run_infer "$PHASE3_DIR" "$OUTPUT_DIR/${BASENAME}_p3_hybrid.wav" \
  "--no_silence_gate --no_pitch_fusion" \
  "P3 hybrid (timbro puro)"

run_infer "$PHASE3_DIR" "$OUTPUT_DIR/${BASENAME}_p3_preset.wav" \
  "--preset $PRESET --no_silence_gate --no_pitch_fusion" \
  "P3 preset $PRESET (timbro puro)"

# ─── Varianti con head attiva (opzionale) ─────────────────────────────────
if [[ "$INCLUDE_HEAD_VARIANTS" == "1" ]]; then
  info "═══ Phase 3 — con silence_gate + pitch_fusion attivi ═══"
  run_infer "$PHASE3_DIR" "$OUTPUT_DIR/${BASENAME}_p3_hybrid_head.wav" \
    "" \
    "P3 hybrid + head"

  run_infer "$PHASE3_DIR" "$OUTPUT_DIR/${BASENAME}_p3_preset_head.wav" \
    "--preset $PRESET" \
    "P3 preset + head"
fi

# ─── Summary ──────────────────────────────────────────────────────────────
echo
ok "Renders completi. Output in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR/${BASENAME}_"*.wav 2>/dev/null | sort
echo
info "Checklist ascolto:"
echo
echo "  TIMBRO (confronto primario):"
echo "    _p1_hybrid vs _p3_hybrid     → decoder P3 ha timbro più pulito di v3?"
echo "    _p1_preset vs _p3_preset     → stessa domanda in preset mode"
echo
echo "  Cosa cercare in P3:"
echo "    + Meno artefatti agli attacchi (distillation guida z verso struttura onset)"
echo "    + Decay più naturale dopo le note"
echo "    + Silenzi più definiti (encoder-z più 'consapevole' delle pause)"
echo "    - Se pitch sbagliato o timbro peggiore → z collapse parziale"
echo "      (in quel caso: abbassa lambda_aux nel gin e riavvia P3)"
echo
if [[ "$INCLUDE_HEAD_VARIANTS" == "1" ]]; then
  echo "  HEAD (confronto secondario):"
  echo "    _p3_hybrid vs _p3_hybrid_head → head runtime aiuta P3?"
  echo "    _p3_preset vs _p3_preset_head → idem preset"
  echo
fi
echo "  Il confronto più importante: _p1_hybrid vs _p3_hybrid."
echo "  Se P3 suona meglio → Phase 3 sta funzionando, continua fino a 100k step."
