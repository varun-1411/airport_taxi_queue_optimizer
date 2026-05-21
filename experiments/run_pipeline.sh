#!/bin/bash
# ================================================================
# Complete experiment pipeline:
#   1. Find initial state (π₀)
#   2. Run Full-Day vs Greedy vs MPC comparison
#
# Usage:
#   bash experiments/run_pipeline.sh               # default settings
#   bash experiments/run_pipeline.sh --fast         # quick test
#   bash experiments/run_pipeline.sh --full         # full scale
# ================================================================

set -e

MODE=${1:---default}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$MODE" = "--fast" ]; then
    echo "=== FAST MODE (testing) ==="
    N_DAYS=3
    N_REFINE=2
    N_SAMPLES=3
    COMMIT=36
    MAX_ITER=100
    LR=1.0
    EPS=0.1
elif [ "$MODE" = "--full" ]; then
    echo "=== FULL SCALE MODE ==="
    N_DAYS=10
    N_REFINE=5
    N_SAMPLES=10
    COMMIT=36
    MAX_ITER=500
    LR=1.0
    EPS=0.1
else
    echo "=== DEFAULT MODE ==="
    N_DAYS=10
    N_REFINE=5
    N_SAMPLES=5
    COMMIT=36
    MAX_ITER=300
    LR=1.0
    EPS=0.1
fi

PI0_DIR="results/initial_state"
COMP_DIR="results/comparison_${TIMESTAMP}"

echo ""
echo "Settings:"
echo "  π₀ calibration: ${N_DAYS} days + ${N_REFINE} refinement rounds"
echo "  Comparison:      ${N_SAMPLES} samples, commit=${COMMIT}"
echo "  Adam:            max_iter=${MAX_ITER}, lr=${LR}"
echo "  Output:          ${PI0_DIR}, ${COMP_DIR}"
echo ""

# ================================================================
# STEP 1: INITIAL STATE CALIBRATION
# ================================================================
echo "========================================"
echo "STEP 1: Find initial distribution π₀"
echo "========================================"

# Check if π₀ already exists
if [ -f "${PI0_DIR}/pi0_optimized.npy" ]; then
    echo "  Found existing π₀ at ${PI0_DIR}/pi0_optimized.npy"
    echo "  Skipping calibration (delete to recalibrate)"
else
    python experiments/find_initial_state.py \
        --n_days $N_DAYS \
        --n_refine $N_REFINE \
        --max_iter $MAX_ITER \
        --lr $LR \
        --epsilon $EPS \
        --sensitivity \
        --out_dir "$PI0_DIR" \
        2>&1 | tee "${PI0_DIR}/log.txt"
fi

PI0_FILE="${PI0_DIR}/pi0_optimized.npy"
echo ""
echo "Using π₀ from: ${PI0_FILE}"

# ================================================================
# STEP 2: COMPARISON (with state sampling)
# ================================================================
echo ""
echo "========================================"
echo "STEP 2: Full-Day vs Greedy vs MPC"
echo "========================================"

mkdir -p "$COMP_DIR"

python experiments/run_optimizers.py \
    --n_samples $N_SAMPLES \
    --commit $COMMIT \
    --max_iter $MAX_ITER \
    --lr $LR \
    --epsilon $EPS \
    --sample_state \
    --pi0 "$PI0_FILE" \
    --out_dir "$COMP_DIR" \
    2>&1 | tee "${COMP_DIR}/log.txt"

# ================================================================
# STEP 3: ALSO RUN WITHOUT SAMPLING (deterministic comparison)
# ================================================================
echo ""
echo "========================================"
echo "STEP 3: Deterministic comparison (no sampling)"
echo "========================================"

DET_DIR="${COMP_DIR}/deterministic"
mkdir -p "$DET_DIR"

python experiments/run_optimizers.py \
    --n_samples 1 \
    --commit $COMMIT \
    --max_iter $MAX_ITER \
    --lr $LR \
    --epsilon $EPS \
    --pi0 "$PI0_FILE" \
    --out_dir "$DET_DIR" \
    2>&1 | tee "${DET_DIR}/log.txt"

# ================================================================
# SUMMARY
# ================================================================
echo ""
echo "========================================"
echo "DONE"
echo "========================================"
echo ""
echo "Results:"
echo "  Initial state:    ${PI0_DIR}/"
echo "    pi0_do_nothing.npy   — day 1 of deployment"
echo "    pi0_optimized.npy    — steady daily operations"
echo "    convergence.png      — convergence plots"
echo "    distributions.png    — π_DN* vs π_OPT* marginals"
echo ""
echo "  Stochastic comparison: ${COMP_DIR}/"
echo "    objectives.png       — boxplot"
echo "    controls.png         — μ⁺, μ⁻ profiles (mean ± std)"
echo "    per_block.png        — per-block cost comparison"
echo "    summary.json         — numbers for the paper"
echo ""
echo "  Deterministic:         ${DET_DIR}/"
echo "    (same files, single run each)"
echo ""

# Print key numbers
echo "Key numbers for the paper:"
python -c "
import json
# Stochastic
with open('${COMP_DIR}/summary.json') as f:
    s = json.load(f)
print(f'  Full-Day:       {s[\"full_day\"]:.2f}')
print(f'  Do-Nothing:     {s[\"do_nothing\"]:.2f}')
print(f'  Greedy (mean):  {s[\"greedy_mean\"]:.2f} ± {s[\"greedy_std\"]:.2f}')
print(f'  MPC (mean):     {s[\"mpc_mean\"]:.2f} ± {s[\"mpc_std\"]:.2f}')
print(f'  Greedy gap:     {(s[\"greedy_mean\"]-s[\"full_day\"])/s[\"full_day\"]*100:+.2f}%')
print(f'  MPC gap:        {(s[\"mpc_mean\"]-s[\"full_day\"])/s[\"full_day\"]*100:+.2f}%')
"