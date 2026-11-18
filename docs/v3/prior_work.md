# What v3 measures that nearby work does not

Status: WP0 draft. This is a check of the three literatures named in `docs/v3/plan.md`, not a systematic review. It names the closest paper in each, and the decision that follows.

v3 asks one measurement question. A learned simulator is trained on trajectories from a process system, then asked a "what if" outside that range: a new disturbance, or a control it did not see. A refusal score ranks those scenarios. At a fixed coverage, the false-accept rate is the share of accepted scenarios whose answer is wrong. Wrong is scored twice: once as large rollout error (`bad_rmse`), once as a missed or invented limit violation (`bad_event`). The score has to beat random, and beat simple baselines chosen only on calibration data, by a margin fixed before the confirmatory sample.

Three papers sit next to that question and measure something else.

1. **Selective prediction.** Yonatan Geifman and Ran El-Yaniv, "Selective Classification for Deep Neural Networks", NeurIPS 2017. A trained classifier abstains on low-confidence inputs, and the paper plots risk against coverage. The unit is one labelled image, the error is a wrong class, and the test inputs come from the same task as training. There is no multi-step simulator, no intervention outside the training range, and no process limit.

2. **Conformal prediction for dynamical models.** Lars Lindemann, Matthew Cleaveland, Gihyun Shim, and George J. Pappas, "Safe Planning in Dynamic Environments Using Conformal Prediction", IEEE Robotics and Automation Letters, 8(8):5116–5123, 2023. Conformal prediction builds a region around a trajectory forecast of another agent, and a planner stays safe if the true trajectory falls inside that region. The guarantee is coverage of the forecast. It is not a test of whether a score refuses the simulator's own counterfactual answer, and it does not split average error from limit-violation error.

3. **Uncertainty in learned dynamics for offline model-based RL.** Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims, "MOReL: Model-Based Offline Reinforcement Learning", NeurIPS 2020. An ensemble of learned dynamics marks state-action pairs where the members disagree, and the policy is sent to a halt state there. Success is the return of the policy trained inside that pessimistic model. The concurrent soft-penalty method is Tianhe Yu et al., "MOPO: Model-based Offline Policy Optimization", NeurIPS 2020. Neither paper reports, scenario by scenario, whether the uncertainty score accepted a simulator answer that broke a process limit.

No one of these is a benchmark with the same measurement, so v3 is not repositioned and is not stopped. v3 is the pre-registered comparison, on a process system no judge has seen, of whether existing refusal signals reduce that false-accept rate for both kinds of wrong answer, at a sample size where one scenario cannot move the rate by more than the decision threshold.
