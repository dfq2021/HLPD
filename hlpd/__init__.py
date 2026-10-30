"""HLPD: Human Language Preference Detection for machine-revised text.

Package layout
--------------
hlpo_loss.py   Human Language Preference Optimization (HLPO) objective and the
               Dynamic-beta scheduler.
model.py       HLPDScorer: LoRA-tuned scoring model + HLP-CPC criterion.
dataset.py     Paired (human, machine-revised) datasets.
engine.py      Training / evaluation loops.
train.py       `python -m hlpd.train`      train (or evaluate) a scorer.
evaluate.py    `python -m hlpd.evaluate`   fast evaluation with an inference checkpoint.
detect.py      `python -m hlpd.detect`     interactive command-line detector.
"""

__version__ = "1.0.0"
