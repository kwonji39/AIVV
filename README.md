# AIVV

AIVV is an anomaly detection pipeline for UUV yaw fault data. It combines an LSTM-based forecasting engine, conformal checks, and LLM-backed agent decisions.

## Setup

Create the conda environment used for this project:

```bash
conda env create -f environment.yml
conda activate anomaly_detection
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
python main.py --epochs 1 --samples 1 --uuv-downsample 100 --window-size 5
```

The default dataset is `data/uuv/UUV_yaw_fault_0_new.txt`. Runtime logs, plots, and checkpoints are generated locally and ignored by git.

## Ablations

```bash
python ablation.py --mode math_sentry_only --epochs 1 --samples 5
python ablation.py --mode no_inspector_tuner --epochs 1 --samples 5
```
