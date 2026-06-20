# Experiment Protocol

The smoke experiment generates normal-policy training trajectories and separate in-distribution and intervention-shift test trajectories. Models are trained only on the normal-policy training split. Refusal judges score each test scenario, and lower-risk scenarios are accepted at fixed coverage levels.

The benchmark reports false accept rate among accepted scenarios, where a scenario is bad when the configured trajectory error exceeds the configured threshold.

