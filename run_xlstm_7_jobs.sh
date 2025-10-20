#!/bin/bash
# Launch 7 skypilot training runs with xlstm architecture and 3 billion timesteps
# Usage: ./run_xlstm_7_jobs.sh [base_run_name]
#
# This script launches 7 independent training runs using the abes_prog_7_xlstm recipe
# with 3 billion timesteps each on skypilot.

set -e

# Get base run name from argument or use default
BASE_NAME="${1:-xlstm_3b}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Total timesteps: 3 billion
TIMESTEPS=3000000000

# Recipe path
RECIPE="experiments.recipes.eval_v_11_1_25.abes_prog_7_xlstm.train"

echo "=========================================="
echo "Launching 7 xlstm training runs"
echo "=========================================="
echo "Base name: ${BASE_NAME}"
echo "Timestamp: ${TIMESTAMP}"
echo "Total timesteps: ${TIMESTEPS}"
echo "Recipe: ${RECIPE}"
echo "=========================================="
echo ""

# Launch 7 jobs
for i in {1..7}; do
    RUN_NAME="${BASE_NAME}_${TIMESTAMP}_run${i}"
    
    echo "Launching job ${i}/7: ${RUN_NAME}"
    
    ./devops/skypilot/launch.py "${RECIPE}" \
        run="${RUN_NAME}" \
        trainer.total_timesteps="${TIMESTEPS}"
    
    echo "✓ Job ${i} launched: ${RUN_NAME}"
    echo ""
    
    # Small delay to avoid overwhelming the system
    sleep 2
done

echo "=========================================="
echo "All 7 jobs launched successfully!"
echo "=========================================="
echo ""
echo "To monitor jobs, use:"
echo "  sky jobs queue"
echo ""
echo "To view logs for a specific job:"
echo "  sky jobs logs <JOB_ID>"
echo ""
echo "Run IDs:"
for i in {1..7}; do
    echo "  - ${BASE_NAME}_${TIMESTAMP}_run${i}"
done

