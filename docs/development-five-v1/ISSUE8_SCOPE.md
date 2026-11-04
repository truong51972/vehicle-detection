# Proposed five-class scope for issue #8

This document scopes the subsequent one-time holdout benchmark. It does not
release reserved pixels or authorize final inference. Issue #8 remains open.

Replace its historical six-class matrix with the five-class target
`[car, truck, bus, motorcycle, bicycle]`. Retain the pinned original COCO-80
pretrained checkpoint and both paired five-class S10 best checkpoints, each on
Raw and frozen Gamma-or-Skip v2 inputs. Pretrained prediction mapping is
`{2:0, 7:1, 5:2, 3:3, 1:4}`; railway predictions/GT are excluded from the
derived target while every reserved image ID remains present.

Use the development-selected S10, native freeze=11, 20-epoch training recipe
and selected best checkpoints. Both branches selected epoch 6 on shared Raw
development data. Retaining both avoids declaring a winner from one seed.
No S20 expansion, policy revision or additional training is selected here.
Exact candidate hashes, development identities and settings belong in the
development candidate lock produced after the six-cell checks.

Before release, bind the 9,730 reserved IDs (after the previously disclosed 270
exclusions), groups, five-class conversion, original protocol/policy identities,
checkpoints, JPEG95 encoding, native inference settings and runtime in a final
protocol lock. Complete the final pixel/geometry/content-leakage checks under
the separate release process. Development export hashes do not stand in for
uncreated final exports. All final cells must share IDs and GT.

Keep natural-distribution metrics, per-class/condition/size support, paired
group uncertainty, failure examples and measured deployment costs in #8.
The development image-presence size slices are not area-restricted AP; decide
and freeze final size semantics before opening holdout. Native mean validation
timings and a small offline CPU sample do not replace #8's synchronized p50/p95,
end-to-end latency or peak-VRAM measurements.

Carry the route analysis forward using the actual frozen export decisions and
matched Raw/Adaptive IDs for each fixed model. Development Gamma covers 3,307
images (23.6637%); 3,280 are night, but another 2,355 night images are Skip.
Night is not a substitute for the brightness-triggered Gamma population, and
these prevalence figures do not predict the reserved population. Verify Skip
JPEG/native-match equivalence before attributing any input delta to Gamma.
The development route deltas are descriptive without route-specific intervals;
freeze support/class-denominator and group-uncertainty rules before making a
robust final route claim. Keep inference-input effects separate from the
single-seed training-branch comparison; macro AP deltas are not additive across
routes. The existing development candidate/checkpoint lock is unchanged.

The separate synthetic robustness matrix requested by #8 still needs an exact
five-class protocol, conditions, severities, seeds and budget before release.
It is not run or silently dropped by issue #22. Freeze that scope before any
final scores, or explicitly defer it as a separately labeled protocol decision.

Do not use final results for checkpoint/policy/scale tuning, rebalance natural
conditions, or claim a five-vs-six macro-mAP improvement. Historical six-class
results remain context; a common-five algorithmic comparison needs comparable
GT/model settings and a separate analysis.
