# Frozen experiment protocol

## Problem and environment

The task is a dynamic flexible job shop scheduling problem with job-release, operation-completion, machine-failure-start, and repair-completion events. A job is hidden from the graph state and action set until its release time. A failed or repairing machine cannot receive a new operation. If operation completion and failure start occur within `1e-5`, they are treated as simultaneous. Completion is recorded first, followed by repair completion, failure start, and job release updates; the next dispatch decision is generated only after the state update.

## Policy and inference

The graph policy is initialized from `checkpoints/base/reference_save_10_5.pt` and fine-tuned with a fixed stability-cost advantage. The main setting uses three training seeds and a fixed multiplier of 1.0 with multiplier learning rate 0.0. Inference generates 20 stochastic candidates. Candidates are first checked for scheduling feasibility, then screened by the stability budget, and finally ranked by makespan. If no feasible candidate satisfies the budget, the feasible candidate with the smallest stability cost is retained.

## Statistical unit

Comparisons use a base-instance × disturbance-scenario replicate as the paired unit. The three independent training runs are averaged within each paired unit. Candidate trajectories are not treated as independent statistical observations. The reported analysis uses paired differences, 95% confidence intervals, Wilcoxon signed-rank tests, paired effect sizes, Holm correction, and clustered bootstrap summaries where specified by the manuscript.
