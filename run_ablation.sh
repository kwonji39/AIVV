#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
# seeds=(20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49)
# seeds=(50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69)
# seeds=(72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89)
# seeds=(92 93 94 95 96 97 98 99)
seeds=(16 16 16 16 16 16 16 16 16)

base_args=(
    --dataset uuv
    --uuv-file data/uuv/UUV_yaw_fault_0_new.txt
    --uuv-downsample 20
    --uuv-train-frac 0.7
    --window-size 10
    --forecast-horizon 2
    --epochs 100
    --fault-persistence 1
    --plot
    --disable-council-temporal-filter
    --mode no_inspector_tuner
)

extra_args=("$@")

batch_stamp="$(date +%Y%m%d_%H%M%S)"
batch_dir="logs/run_batch_${batch_stamp}"
mkdir -p "$batch_dir"

echo "Ablation batch run directory: $batch_dir"

for ((i=1; i<=${#seeds[@]}; i++)); do
    seed="${seeds[$i]}"
    seed_run_dir="$batch_dir/seed_${seed}_run${i}"

    echo "========================================"
    echo "Running ABLATION with seed $seed (run $i)"
    echo "Log dir: $seed_run_dir"
    echo "========================================"
    python ablation.py "${base_args[@]}" --seed "$seed" --run-log-dir "$seed_run_dir" "${extra_args[@]}"
done
