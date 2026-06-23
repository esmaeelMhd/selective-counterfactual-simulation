# Public Release Exposure Decision

## Selected route

Option B

## Reason

Use a clean public mirror for the first public release. The repository has multiple research tags and branches, including v2 diagnostic artifacts, so publishing the current repository history directly is unnecessary risk for the first public benchmark prototype.

## Branches checked

- local: main
- local: v2-scientific-strengthening
- local: public-release-v1
- remote: origin/main

## Tags checked

- v0-claim-downgraded
- v0-controlled-expansion-diagnostic
- v0-evidence-audit
- v1-calibrated-refusal-judge
- v1-calibrated-refusal-judge-improved
- v1-cstr-frozen-protocol-replication
- v1-cstr-weakness-diagnosis
- v1-current-status-sync
- v1-effect-size-practical-utility-audit
- v1-limitations-first-technical-note
- v1-repair-signal-semantics
- v1.1-benchmark-usability
- v1.2-public-benchmark-strengthening
- v2-scientific-strengthening
- v2-underperformance-diagnosis

## History risk

Current-repository history should not be made public as the first release without a separate full history audit. The selected route avoids exposing historical branches and tags.

## Private data risk

Use the public release audit script before mirroring. The clean mirror should be created from the accepted release snapshot only.

## Large artifact risk

Tracked files over 20 MB are reviewed in `reports/public_release_artifact_cleanup.md`. The clean mirror route avoids publishing ignored local environments and caches.

## Final decision

PUBLIC_RELEASE_ROUTE_ACCEPTED
