#!/bin/bash
# Interactive command-line detector (~11 GB GPU memory; add --device cpu to run without a GPU).
# --scenario one of: all | polish | rewrite | expand | generate | adversarial
python -m hlpd.detect \
    --checkpoint models/HLPD-inference \
    --ref_path local_infer_ref \
    --scenario all \
    --detail
