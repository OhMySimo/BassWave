#!/usr/bin/env bash
# ab_listening_test.sh — A/B render con e senza pitch fusion.
#
# Output: fino a 8 file WAV per diagnostica.
#   <basename>_p1_hybrid.wav            P1, z dall'input
#   <basename>_p1_preset.wav            P1, z preset
#   <basename>_p2_hybrid.wav            P2, z input, fusion ON
#   <basename>_p2_preset.wav            P2, z preset, fusion ON
#   <basename>_p2_hybrid_nofusion.wav   P2, z input, fusion OFF  (diagnostico)
#   <basename>_p2_preset_nofusion.wav   P2, z preset, fusion OFF (diagnostico)
# Con INCLUDE_NOGATE=1 aggiunge anche:
#   <basename>_p2_hybrid_nogate.wav     P2, gate OFF
#   <basename>_p2_preset_nogate.wav     P2, gate OFF

set -euo pipefail

PHASE1_DIR="${PHASE1_DIR:-/media/simone/NVME/runs/basswave_v4_phase1}"
PHASE2_DIR="${PHASE2_DIR:-/media/simone/NVME/runs/basswave_v4_phase2}"
TFRECORD_PATTERN="${TFRECORD_PATTERN:-/media/simone/NVME/MidiDataset/BassWave_TFR/basswave-train-*.tfrecord}"
PRESET="${PRESET:-Modern}"
N_PRESET_SAMPLES="${N_PRESET_SAMPLES:-64}"

INPUT="${1:-}"
OUTPUT_DIR="${2:-./ab_test}"
INCLUDE_NOFUSION="${INCLUDE_NOFUSION:-1}"  # default ON: importante per diagnosi
INCLUDE_NOGATE="${INCLUDE_NOGATE:-0}"

if [[ -z "$INPUT" ]]; then
  cat <<EOF
Usage: $0 <input.wav> [output_dir]

Variabili d'ambiente opzionali:
  PHASE1_DIR        (default: /media/simone/NVME/runs/basswave_v4_phase1)
  PHASE2_DIR        (default: /media/simone/NVME/runs/basswave_v4_phase2)
  TFRECORD_PATTERN  (default: BassWave_TFR/basswave-train-*.tfrecord)
  PRESET            (default: Modern)
  N_PRESET_SAMPLES  (default: 64)
  INCLUDE_NOFUSION  (default: 1; 0 per skippare A/B su pitch fusion)
  INCLUDE_NOGATE    (default: 0; 1 per A/B su silence gate)

Esempi:
  $0 /tmp/bass.wav ./ab_test_modern
  PRESET=Funk_Fusion $0 input.wav ./ab_funk
  INCLUDE_NOGATE=1 $0 input.wav ./ab_full
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
        $extra_args 2>&1 | tail -5; then
    if [[ -f "$out_path" ]]; then
      ok "$out_path"
    else
      warn "Done but file MISSING: $out_path"
    fi
  else
    warn "FAILED: $label (vedi output sopra)"
  fi
  echo
}

# ─── Phase 1 ──────────────────────────────────────────────────────────────
info "═══ Phase 1: $PHASE1_DIR ═══"
run_infer "$PHASE1_DIR" "$OUTPUT_DIR/${BASENAME}_p1_hybrid.wav" \
  "" "P1 hybrid"
run_infer "$PHASE1_DIR" "$OUTPUT_DIR/${BASENAME}_p1_preset.wav" \
  "--preset $PRESET" "P1 preset ($PRESET)"

# ─── Phase 2 (fusion ON) ──────────────────────────────────────────────────
info "═══ Phase 2: $PHASE2_DIR ═══"
run_infer "$PHASE2_DIR" "$OUTPUT_DIR/${BASENAME}_p2_hybrid.wav" \
  "" "P2 hybrid (fusion ON)"
run_infer "$PHASE2_DIR" "$OUTPUT_DIR/${BASENAME}_p2_preset.wav" \
  "--preset $PRESET" "P2 preset (fusion ON)"

# ─── Phase 2 no pitch fusion (diagnostico octave) ────────────────────────
if [[ "$INCLUDE_NOFUSION" == "1" ]]; then
  info "═══ Phase 2 senza pitch fusion (diagnostico octave) ═══"
  run_infer "$PHASE2_DIR" "$OUTPUT_DIR/${BASENAME}_p2_hybrid_nofusion.wav" \
    "--no_pitch_fusion" "P2 hybrid (fusion OFF)"
  run_infer "$PHASE2_DIR" "$OUTPUT_DIR/${BASENAME}_p2_preset_nofusion.wav" \
    "--preset $PRESET --no_pitch_fusion" "P2 preset (fusion OFF)"
fi

# ─── Phase 2 no gate (diagnostico) ───────────────────────────────────────
if [[ "$INCLUDE_NOGATE" == "1" ]]; then
  info "═══ Phase 2 senza silence gate ═══"
  run_infer "$PHASE2_DIR" "$OUTPUT_DIR/${BASENAME}_p2_hybrid_nogate.wav" \
    "--no_silence_gate" "P2 hybrid no-gate"
  run_infer "$PHASE2_DIR" "$OUTPUT_DIR/${BASENAME}_p2_preset_nogate.wav" \
    "--preset $PRESET --no_silence_gate" "P2 preset no-gate"
fi

# ─── Summary ──────────────────────────────────────────────────────────────
echo
ok "Renders completi. Output in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR/${BASENAME}_"*.wav 2>/dev/null
echo
info "Diagnostica:"
echo "  → octave error in P2 hybrid? Confronta:"
echo "      ${BASENAME}_p2_hybrid.wav            (fusion ON)"
echo "      ${BASENAME}_p2_hybrid_nofusion.wav   (fusion OFF)"
echo "    Se nofusion ha pitch corretto → fusion troppo aggressiva."
echo
echo "  → P2 preset crashato? Fix applicato: il nuovo infer inietta midi_cond"
echo "    in features prima del decode. Se ancora crasha, mandami il log COMPLETO."
