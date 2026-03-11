#!/bin/bash
# run server.sh first

python infer_real.py \
  --host 0.0.0.0 \
  --port 8000 \
  --prompt "prompt" \
  --hz 5 \
  --max_steps 500