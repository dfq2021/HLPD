"""Turn an HLPD score into a calibrated probability.

The HLP-CPC score is threshold-free; for the demos we estimate
P(machine-revised | score) non-parametrically from the score distributions of
our test sets (``local_infer_ref/<scenario>/*.hlpd.json``): among the ``k``
reference passages whose scores are closest to the query score, the fraction
that is machine-revised is returned (same estimator as Fast-DetectGPT / ImBD).
"""
import glob
import json
import os

import numpy as np

SCENARIOS = ["polish", "rewrite", "expand", "generate", "adversarial"]
SCENARIO_LABELS = {
    "polish": "machine-polished",
    "rewrite": "machine-rewritten",
    "expand": "machine-expanded",
    "generate": "machine-generated",
    "adversarial": "adversarially multi-task revised",
}


class ProbEstimator:
    def __init__(self, ref_path="./local_infer_ref", scenarios=None, k=100):
        self.scenarios = scenarios or SCENARIOS
        self.k = k
        self.real, self.fake = {}, {}
        for s in self.scenarios:
            real, fake = [], []
            for f in glob.glob(os.path.join(ref_path, s, "*.json")):
                with open(f) as fin:
                    res = json.load(fin)
                real.extend(res["crit_real"])
                fake.extend(res["crit_sampled"])
            self.real[s] = np.asarray(real, dtype=np.float64)
            self.fake[s] = np.asarray(fake, dtype=np.float64)
        self.available = [s for s in self.scenarios if len(self.real[s]) > 0]
        total = sum(len(self.real[s]) + len(self.fake[s]) for s in self.available)
        print(f"ProbEstimator: {total} reference scores from {len(self.available)} scenario(s): {self.available}")

    def _prob(self, real, fake, score):
        if len(real) == 0 or len(fake) == 0:
            return float("nan")
        both = np.concatenate([real, fake])
        k = min(self.k, max(10, len(both) // 10))
        offset = np.sort(np.abs(both - score))[min(k, len(both) - 1)]
        cnt_real = np.sum((real > score - offset) & (real < score + offset))
        cnt_fake = np.sum((fake > score - offset) & (fake < score + offset))
        return float(cnt_fake / max(1, cnt_real + cnt_fake))

    def prob(self, score, scenario="all"):
        """P(machine-revised | score) for one scenario, or pooled over all."""
        if scenario == "all":
            real = np.concatenate([self.real[s] for s in self.available])
            fake = np.concatenate([self.fake[s] for s in self.available])
            return self._prob(real, fake, score)
        return self._prob(self.real[scenario], self.fake[scenario], score)

    def prob_detail(self, score):
        """Per-scenario probabilities as ``{scenario: prob}``."""
        return {s: self._prob(self.real[s], self.fake[s], score) for s in self.available}

    def percentile(self, score, scenario="all"):
        """Where the score falls within the human / machine reference distributions."""
        scen = self.available if scenario == "all" else [scenario]
        real = np.concatenate([self.real[s] for s in scen])
        fake = np.concatenate([self.fake[s] for s in scen])
        return float((real < score).mean()), float((fake < score).mean())
