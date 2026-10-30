"""Human Language Preference Optimization (HLPO) objective.

Given a pair (x_h, x_m) -- a human-written passage and its machine-revised
version -- HLPO trains the scoring model p_theta to *prefer human writing*:

    r(x_h, x_m) = [log p_theta(x_h) - log p_theta(x_m)]
                - [log p_ref(x_h)   - log p_ref(x_m)]              (Eq. 6)

Two objectives are implemented (Sec. 2.2 / Table 4 of the paper):

* ``sigmoid``  -- the DPO / ImBD-style objective   -log sigma(beta * r)
* ``linear``   -- the HLPO objective                -beta_t * r        (Eq. 5)

``beta`` is either fixed or scheduled on-the-fly by the variance-aware
*Dynamic-beta* scheduler (Eq. 7): beta grows while the margin variance
inside a sliding window is low (confident model) and shrinks when it is
high (noisy signal).

Notation used in the paper:  ``‡`` = linear loss, ``⋆`` = Dynamic-beta.
The full method (HLPO‡⋆) is ``loss_type="linear", dynamic_beta=True``.
"""
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


class DynamicBetaScheduler:
    """Variance-aware beta scheduler (Dynamic-beta, Eq. 7).

    The reward margins ``r`` of the most recent ``window_size`` training
    samples are kept in a sliding window.  Their variance is smoothed with
    an exponential moving average whose factor ``rho`` is itself adapted
    (lowered when the variance fluctuates strongly so that the EMA reacts
    faster, raised otherwise).  The smoothed variance is compared with two
    thresholds:

    * ``ema_var < low_var``   ->  beta += beta_up    (capped at ``beta_max``)
    * ``ema_var > high_var``  ->  beta -= beta_down  (floored at ``beta_min``)

    Note: the window is taken over *samples across steps*, so the scheduler is
    well defined for batch size 1 (the setting used in all our experiments).
    """

    def __init__(self, beta_initial=0.05, beta_max=2.0, beta_min=0.1,
                 window_size=10, low_var=0.5, high_var=0.7,
                 rho_init=0.9, rho_min=0.8, rho_max=0.99, rho_step=0.01,
                 beta_up=0.02, beta_down=0.05):
        self.beta = beta_initial
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.low_var = low_var
        self.high_var = high_var
        self.rho = rho_init
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_step = rho_step
        self.beta_up = beta_up
        self.beta_down = beta_down
        self.window = deque(maxlen=window_size)   # recent margins
        self.var_history = deque(maxlen=window_size)  # recent window variances
        self.ema_var = None

    def _window_variance(self):
        if len(self.window) < 2:
            return None
        return float(np.var(np.asarray(self.window, dtype=np.float64), ddof=1))

    @torch.no_grad()
    def step(self, margin: torch.Tensor) -> float:
        for m in margin.detach().float().flatten().tolist():
            self.window.append(m)
        curr_var = self._window_variance()
        if curr_var is None:            # not enough samples yet
            return self.beta
        self.var_history.append(curr_var)
        hist_avg = float(np.mean(self.var_history))

        # adapt the EMA smoothing factor to the amount of fluctuation
        if self.ema_var is not None:
            deviation = abs(curr_var - hist_avg)
            if deviation > 0.5 * hist_avg:
                self.rho = max(self.rho_min, self.rho - self.rho_step)
            else:
                self.rho = min(self.rho_max, self.rho + self.rho_step)

        self.ema_var = curr_var if self.ema_var is None else \
            self.rho * self.ema_var + (1.0 - self.rho) * curr_var

        # double-threshold update of beta
        if self.ema_var < self.low_var:
            self.beta = min(self.beta_max, self.beta + self.beta_up)
        elif self.ema_var > self.high_var:
            self.beta = max(self.beta_min, self.beta - self.beta_down)
        return self.beta

    def state_dict(self):
        return {"beta": self.beta, "rho": self.rho, "ema_var": self.ema_var,
                "window": list(self.window)}


class WindowBetaScheduler(DynamicBetaScheduler):
    """Simpler alternative kept for ablations: the sliding-window variance is
    used directly (no EMA smoothing)."""

    def __init__(self, beta_initial=0.05, beta_max=2.0, beta_min=0.1,
                 window_size=30, low_var=0.3, high_var=1.0, **kw):
        super().__init__(beta_initial, beta_max, beta_min, window_size, low_var, high_var, **kw)

    @torch.no_grad()
    def step(self, margin: torch.Tensor) -> float:
        for m in margin.detach().float().flatten().tolist():
            self.window.append(m)
        var = self._window_variance()
        if var is None:
            return self.beta
        if var < self.low_var:
            self.beta = min(self.beta_max, self.beta + self.beta_up)
        elif var > self.high_var:
            self.beta = max(self.beta_min, self.beta - self.beta_down)
        return self.beta


class HLPOLoss:
    """HLPO objective (callable).

    Parameters
    ----------
    beta : float
        Initial (or fixed) temperature ``beta``.
    loss_type : {"linear", "sigmoid"}
        ``linear`` is the HLPO objective (Eq. 5); ``sigmoid`` is the
        DPO / ImBD objective kept for ablation.
    dynamic_beta : bool
        Enable the Dynamic-beta scheduler (Eq. 7).
    scheduler : {"ema", "window"}
        Scheduler flavour used when ``dynamic_beta`` is on.
    """

    def __init__(self, beta=0.05, loss_type="linear", dynamic_beta=True, scheduler="ema", **scheduler_kwargs):
        assert loss_type in ("linear", "sigmoid"), loss_type
        self.beta = beta
        self.loss_type = loss_type
        self.dynamic_beta = dynamic_beta
        if dynamic_beta:
            cls = DynamicBetaScheduler if scheduler == "ema" else WindowBetaScheduler
            self.scheduler = cls(beta_initial=beta, **scheduler_kwargs)
        else:
            self.scheduler = None

    @property
    def current_beta(self) -> float:
        return self.scheduler.beta if self.scheduler is not None else self.beta

    def __call__(self, model_pref_logprob, model_dispref_logprob, ref_pref_logprob, ref_dispref_logprob,
                 update_beta=True):
        """All inputs are ``[batch]`` tensors of *sequence* log-probabilities
        (human text = preferred, machine-revised text = dis-preferred).
        ``update_beta=False`` (evaluation) leaves the Dynamic-beta state untouched.

        Returns ``(loss, pref_rel, dispref_rel, reward_acc, reward_margin)``;
        the last four are detached scalars for logging.
        """
        pref_rel = model_pref_logprob - ref_pref_logprob           # log p_theta(x_h) - log p_ref(x_h)
        dispref_rel = model_dispref_logprob - ref_dispref_logprob  # log p_theta(x_m) - log p_ref(x_m)
        margin = pref_rel - dispref_rel                            # r(x_h, x_m), Eq. 6

        reward_acc = (pref_rel > dispref_rel).float().mean()
        reward_margin = margin.mean()

        if self.scheduler is not None:
            beta = self.scheduler.step(margin) if update_beta else self.scheduler.beta
        else:
            beta = self.beta

        if self.loss_type == "sigmoid":
            loss = -F.logsigmoid(beta * margin).mean()   # DPO / ImBD
        else:
            loss = -beta * margin.mean()                 # HLPO, Eq. 5

        return loss, pref_rel.mean().detach(), dispref_rel.mean().detach(), reward_acc.detach(), reward_margin.detach()
