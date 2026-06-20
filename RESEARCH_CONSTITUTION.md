# Research Constitution

## Research question

Can a simulator identify which counterfactual intervention scenarios it can answer reliably and abstain on the rest?

## Primary metric

False Accept Rate at fixed coverage.

## Primary claim

A combined refusal judge reduces false acceptance compared with simple uncertainty-only, support-only, and disagreement-only judges.

## Current v0 status

The v0 evidence audit did not support the primary claim. Claim audit and seed robustness were `NOT_SUPPORTED`, while the severity sweep was `MEANINGFUL`. The current `combined_linear` judge is an exploratory baseline, not a supported refusal method.

## Non-claims

- This is not a universal simulator.
- This is not safety certification.
- This is not a plant-wide digital twin.
- This is not autonomous control.
- This is not proof of causal effect from observational data.
- This is not product-ready.

## Kill criterion

If the combined judge does not beat the strongest single-signal judge on false accept rate at matched coverage on at least two systems after v1 experiments, the claim is downgraded or killed.
