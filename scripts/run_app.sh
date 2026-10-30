#!/bin/bash
# Web demo (Gradio) at http://127.0.0.1:7860 . Add --share for a public link, --mock to preview the UI without a model.
python app/app.py \
    --checkpoint models/HLPD-inference \
    --ref_path local_infer_ref \
    --port 7860 "$@"
