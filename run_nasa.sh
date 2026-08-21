#!/usr/bin/env bash
# Experiment 2: AIVV on NASA SMAP/MSL spacecraft telemetry.
#
# Data: the public SMAP/MSL anomaly corpus of Hundman et al. (KDD 2018).
# The original telemanom S3 link is no longer served; the corpus is
# distributed via Kaggle (patrickfleith/nasa-anomaly-detection-dataset-smap-msl).
# Place the corpus test arrays at data/sat_anomaly_detection/test/<CHANNEL>.npy.
#
# Channels and fault-persistence settings as reported in the paper:
#   E-8 -> 10, F-5 -> 5, D-1 -> 4, T-4 -> 1
# Each channel runs at full rate (no downsampling) with a chronological
# 50/50 train/test split, 20 epochs, 20 seeds.
set -euo pipefail

declare -A PERSIST=( ["E-8"]=10 ["F-5"]=5 ["D-1"]=4 ["T-4"]=1 )

for chan in E-8 F-5 D-1 T-4; do
  for seed in {0..19}; do
    echo "======================================="
    echo "Running ${chan} with seed ${seed}"
    echo "======================================="
    python main.py \
      --dataset uuv \
      --uuv-file "data/sat_anomaly_detection/test/${chan}.npy" \
      --uuv-downsample 1 \
      --uuv-train-frac 0.5 \
      --window-size 10 \
      --forecast-horizon 2 \
      --epochs 20 \
      --fault-persistence "${PERSIST[$chan]}" \
      --plot \
      --disable-council-temporal-filter \
      --seed "${seed}" \
      --run-log-dir "logs/sat_batch/${chan}_seed${seed}"
  done
done
