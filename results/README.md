# Results

`results/paper/` holds the raw per-passage scores (`crit_real`, `crit_sampled`), ROC / PR
curves and AUROC of the runs reported in the paper, one `*.hlpd.json` per evaluation file.
`crit_*` are HLPD scores (`−HLP-CPC`, larger = more likely machine-revised).

```
results/paper/
├── adversarial_multitask/{gpt-4o,claude-3.5-sonnet,deepseek-r1,gemini-2.5-pro}/{xsum,writing,squad,wiki}.hlpd.json   Table 3
├── advanced_generation/{claude-3.5-sonnet,deepseek-r1,gemini-2.5-pro,grok-3}/xsum.hlpd.json                        Table 2
├── single_task/{rewrite,polish,expand,generation}/<source model>/<dataset>.hlpd.json                                Tables 1, 5, 7, 9–11
├── multi_seed_adversarial_gemini-2.5-pro/<dataset>/<dataset>_seed<S>.hlpd.json   five seeds (42 199 410 2231 2533), Fig. 4–5
└── baselines/
    ├── imbd_adversarial_multitask/…      ImBD (SPO, GPT-Neo-2.7B) on the adversarial benchmark
    └── gptzero_adversarial_multitask/…   GPTZero API on the GPT-4o adversarial benchmark
```

Print tables with

```bash
python tools/show_results.py --result_path results/paper/adversarial_multitask --table
python tools/show_results.py --result_path results/paper/single_task/rewrite --table
python tools/plot_roc_ci.py --runs results/paper/multi_seed_adversarial_gemini-2.5-pro --output figures/roc_ci.pdf
```

Runs you launch with `scripts/eval_*.sh` are written next to this folder
(`results/<benchmark>/…`, git-ignored); `scripts/eval_all.sh` writes to `evaluation/`.
