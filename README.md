# AIVV

Agent-Integrated Verification and Validation (AIVV) is a hybrid anomaly detection and V&V framework for control-system time-series data. Deep learning models can detect abnormal patterns, but they often struggle to classify whether a mathematically flagged anomaly is a true fault or a nuisance event caused by noise or large transient control responses. That limitation keeps fault validation and downstream Verification and Validation (V&V) dependent on Human-in-the-Loop analysis, which does not scale well across diverse control systems.

AIVV automates this oversight by combining an LSTM/conformal anomaly detector with a deliberative Large Language Model (LLM) outer loop. Mathematically flagged anomalies are escalated to role-specialized council agents that validate nuisance faults and true failures against natural-language requirements. Once a fault is validated, the council assesses post-fault behavior against operational tolerances and produces actionable V&V artifacts, including gain-tuning proposals. The current experiments use an Unmanned Underwater Vehicle (UUV) yaw-fault time-series simulator to demonstrate scalable LLM-mediated oversight for time-series V&V workflows.

## Setup

Create the conda environment used for this project:

```bash
conda env create -f environment.yml
conda activate aivv
```

Or install into an existing environment:

```bash
pip install -r requirements.txt
```

## API Key

LLM-backed modes require a Groq/OpenAI-compatible API key. The code reads it from `GROQ_API_KEY` first, then `GENAI_API_KEY`.

```bash
export GROQ_API_KEY="your_api_key_here"
```

You can also copy `.env.example` to `.env` for local use. `.env` files are ignored by git.

## Quick Run

```bash
python main.py \
  --dataset uuv \
  --uuv-file data/uuv/UUV_yaw_fault_0_new.txt \
  --uuv-downsample 20 \
  --uuv-train-frac 0.7 \
  --window-size 10 \
  --forecast-horizon 2 \
  --epochs 100 \
  --fault-persistence 1 \
  --plot \
  --disable-council-temporal-filter
```

The default dataset is `data/uuv/UUV_yaw_fault_0_new.txt`. Runtime logs, plots, and checkpoints are generated locally and ignored by git.

## Ablations

```bash
python ablation.py \
  --mode math_sentry_only \
  --dataset uuv \
  --uuv-file data/uuv/UUV_yaw_fault_0_new.txt \
  --uuv-downsample 20 \
  --uuv-train-frac 0.7 \
  --window-size 10 \
  --forecast-horizon 2 \
  --epochs 100 \
  --fault-persistence 1 \
  --plot \
  --disable-council-temporal-filter

python ablation.py \
  --mode no_inspector_tuner \
  --dataset uuv \
  --uuv-file data/uuv/UUV_yaw_fault_0_new.txt \
  --uuv-downsample 20 \
  --uuv-train-frac 0.7 \
  --window-size 10 \
  --forecast-horizon 2 \
  --epochs 100 \
  --fault-persistence 1 \
  --plot \
  --disable-council-temporal-filter

python ablation.py \
  --mode full_aivv \
  --dataset uuv \
  --uuv-file data/uuv/UUV_yaw_fault_0_new.txt \
  --uuv-downsample 20 \
  --uuv-train-frac 0.7 \
  --window-size 10 \
  --forecast-horizon 2 \
  --epochs 100 \
  --fault-persistence 1 \
  --plot \
  --disable-council-temporal-filter
```
