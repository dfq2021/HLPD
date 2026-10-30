# Reference score distributions for the demos

`hlpd.detect` and `app/app.py` convert an HLPD score into a probability by looking at
the scores of our test sets (`crit_real` = human passages, `crit_sampled` = machine
passages). Each sub-directory is one calibration *scenario*:

| directory | source | reference sets |
|---|---|---|
| `polish/`, `rewrite/`, `expand/` | `results/paper/single_task/` | GPT-3.5-turbo and GPT-4o revisions of XSum / Writing / SQuAD / PubMed / Wiki |
| `generate/` | `results/paper/single_task/generation/`, `results/paper/advanced_generation/` | GPT-3.5 / GPT-4o / Claude-3.5 / DeepSeek-R1 / Gemini-2.5-Pro / Grok-3 |
| `adversarial/` | `results/paper/adversarial_multitask/` | GPT-4o / Claude-3.5 / DeepSeek-R1 / Gemini-2.5-Pro, four domains |

The files are ordinary result files (`*.hlpd.json`); drop the outputs of your own
`python -m hlpd.evaluate` runs here to re-calibrate for a new domain.
