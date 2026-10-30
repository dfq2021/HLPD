# Third-party notices

HLPD builds on the code and data released by the following projects. Their
licenses apply to the corresponding files.

| Files | Origin | License |
|---|---|---|
| `other_models/baselines.py`, `custom_datasets.py`, `data_builder.py`, `detect_gpt.py`, `detect_llm.py`, `dna_gpt.py`, `fast_detect_gpt.py`, `metrics.py`, `model.py`, `supervised.py` | [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) (Bao et al., 2023) as redistributed by [ImBD](https://github.com/Jiaqi-Chen-00/ImBD) | MIT (Fast-DetectGPT) / Apache-2.0 (ImBD) |
| `ablation_exp/train_gpt_neo_{sft,rlhf,orpo}.py` (preference direction made configurable) | [ImBD](https://github.com/Jiaqi-Chen-00/ImBD) (Chen et al., AAAI 2025) | Apache-2.0 — see `third_party/ImBD_LICENSE` |
| `tools/data_builder.py`, `tools/truncation.py`, `tools/count_words.py`, `tools/solve_llama_output.py` | ImBD | Apache-2.0 |
| `data/train/imbd_polish_{500,3000}.raw_data.json`, `data/single_task/**` (six source models), `data/truncation/polish/gpt-*`, `data/prompts/{polish,expand}_prompts_imbd.json` | ImBD benchmark data (human sources: XSum, SQuAD, WritingPrompts, PubMedQA, WikiText) | ImBD data terms |
| `hlpd/model.py::get_sampling_discrepancy_analytic` | analytic conditional probability curvature of Fast-DetectGPT | MIT |

The HLPD-specific code (`hlpd/`, `app/`, `tools/adversarial_prompt_generator.py`,
`tools/build_revision_data.py`, `tools/build_generation_data.py`,
`tools/humanize.py`, `tools/plot_roc_ci.py`, `tools/show_results.py`,
`other_models/gptzero.py`) and the HLPD data (`data/train/train_multitask_revision.raw_data.json`,
`data/adversarial_multitask/`, `data/advanced_generation/`, `data/multilingual/`,
`data/iterative_revision/`, `data/truncation/adversarial_multitask/`,
`data/prompts/adversarial_prompts.json`) are released under the MIT License.
