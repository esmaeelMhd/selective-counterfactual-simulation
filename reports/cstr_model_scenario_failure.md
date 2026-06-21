# CSTR Model and Scenario Failure Audit

## Question

Are CSTR false accepts concentrated in one model or scenario type?

## Model failure summary

| model | accepted_false_accept_count | accepted_false_accept_rate | mean_accepted_bad_rmse |
| --- | ---: | ---: | ---: |
| hold_last | 105 | 1.000000 | 3.744395 |
| linear_narx | 15 | 0.142857 | 1.015981 |
| mlp_state_space | 78 | 0.742857 | 2.170424 |

## Scenario failure summary

| scenario_type | accepted_false_accept_count | accepted_false_accept_rate | mean_accepted_bad_rmse |
| --- | ---: | ---: | ---: |
| combined_feed_and_cooling_shift | 30 | 0.666667 | 3.124460 |
| cooling_step_change | 30 | 0.666667 | 2.317158 |
| feed_concentration_spike | 30 | 0.666667 | 1.451992 |
| feed_temperature_spike | 30 | 0.666667 | 1.275833 |
| id | 15 | 0.333333 | 1.848434 |
| reaction_rate_shift | 18 | 0.400000 | 1.526042 |
| unsafe_temperature_event | 45 | 1.000000 | 6.164792 |

## Failure concentration

Dominant model: hold_last; share 0.530303.
Dominant scenario type: unsafe_temperature_event; share 0.227273.

## Interpretation

top model share=0.530303, top scenario share=0.227273

## Verdict

DIFFUSE_CSTR_FAILURE
