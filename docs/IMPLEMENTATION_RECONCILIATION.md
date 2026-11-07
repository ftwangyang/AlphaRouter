# Implementation Reconciliation

## Purpose and scope

This note reconciles three distinct objects:

1. the method specified in the manuscript's Eqs. 6–18 and Appendix B, including Eqs. 20–26;
2. the repository's initial tracked snapshot, commit 1c62c3f;
3. the current paper-equation implementation.

It documents implementation changes and evidence boundaries. It does not revise the manuscript, authenticate the origin of previously reported numbers, or claim that those numbers have been reproduced.

The comparison below was made from source code and the supplied manuscript. The repository does not contain the research NPZ panels, original checkpoints, immutable run manifests, or raw result artifacts needed to trace the manuscript tables to a particular execution.

## Paper-aligned target protocol

### Candidate experts

All dates use the same four expert definitions: slow sparse, defensive dense, fast sparse, and adaptive dense. Sparse experts construct the ranked preference in Eq. 6 and project it onto the capped simplex in Eq. 7. Dense experts solve the capped-simplex mean–variance program in Eq. 8 using the shrunk covariance in Eq. 9.

Expert state, rebalance phase, and portfolio drift follow one chronological path rooted at the beginning of the panel. A validation or test slice must not restart an expert at each calendar year.

The router's trailing expert-return channels use the portfolio that each expert actually held on each historical date. The current decision-date portfolio is not applied retrospectively to the whole lookback window.

Each market-state token also contains descriptors calculated from information observable on that token's historical date. Long-horizon descriptors are no longer calculated once at the decision date and repeated across all past tokens.

### Dynamic score and training loss

For expert k and horizon h, the implementation target is

~~~text
S(k) = G(k) - 0.25 D(k) - 0.10 M(k) - 0.0025 T(k) - 0.002 C(k)
~~~

The components follow Appendix B.3:

- G is product(1 + a) - 1, using pre-cost expert portfolio returns a.
- D is sqrt(mean(min(a, 0)^2)); it is not multiplied by sqrt(h).
- M is maximum drawdown of the gross-return wealth path, including the initial wealth V0 = 1 in each running peak.
- T is the sum of daily L1 distances between each target portfolio and the preceding drifted allocation. The first labeled date of a split uses the preceding expert allocation from the continuous path, drifted by the preceding return.
- C is the horizon mean of the raw Herfindahl concentration, sum of squared asset weights.

Transaction cost is the separate cT term in Eq. 11. It is not inserted into G, D, or M.

The raw four-expert scores are converted directly to

~~~text
q = softmax(S / 0.75)
~~~

without date-wise standardization. For router probabilities p, the optimized loss is

~~~text
L = -sum(q * log(p)) + 0.25 * (max(S) - sum(p * S))
~~~

which is the soft-target cross-entropy plus regret in Eqs. 12–13.

### Reference, confidence, voting, and blend

The reference expert is chosen independently for each market from the complete training history:

~~~text
B(k) = ASR(k) + 0.25 CR(k) + 0.10 APY(k)
reference = argmax B(k)
~~~

For every router and decision date, confidence is the difference between its largest and second-largest probabilities. The validation search uses exactly:

~~~text
[0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
~~~

If the difference is below the selected threshold, that router votes for the reference expert; otherwise it votes for its probability argmax. The committee selects the largest vote count. This is plurality selection and does not require an absolute majority. A vote-count tie is resolved by the largest ensemble-mean probability among the tied experts. An exact remaining tie uses the lowest expert index for deterministic execution.

The adaptive-share grid is:

~~~text
[0.000, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200]
~~~

The executed gate and portfolio are the convex blends

~~~text
g = (1 - alpha) * reference_one_hot + alpha * routed_one_hot
w = sum(g(k) * w(k))
~~~

The reference fallback is therefore a required part of Eqs. 14–18, not dead code or an ablation that should be removed during cleanup.

## Reconciliation with initial commit 1c62c3f

| Area | Initial tracked behavior | Current paper-equation behavior | Why it matters |
| --- | --- | --- | --- |
| Sparse and dense experts | Dense portfolios used a short projected-gradient approximation and an additional mean-scaling coefficient; test expert paths restarted each calendar year. | Sparse preferences are projected as in Eqs. 6–7; dense portfolios use a constrained numerical solve for Eqs. 8–9; every split is sliced from one chronological expert path. | Candidate portfolios now match the stated optimization and do not gain test-only annual resets. |
| Router expert evidence | The current decision-date portfolio was applied retrospectively to every return in its trailing window. | Each trailing date uses that date's causal expert portfolio and realized return. | Router inputs now describe the historical expert path rather than a hindsight, constant-weight counterfactual. |
| Dynamic-score path | Costs were folded into daily returns before log growth, downside, and drawdown were computed. Growth was a sum of log returns. | G, D, and M use pre-cost gross portfolio returns; the separate 0.0025 T penalty is subtracted once. G is cumulative simple return. | This aligns the score decomposition with Eqs. 11 and 20–25. |
| Turnover at label boundaries | The first target inside each split was measured from an equal-weight initialization. | The score path carries the preceding expert allocation across the boundary, applies the preceding return, and measures the first target against that drifted allocation. | Eq. 25 now uses the same causal drift convention on the first labeled validation date as on later dates. |
| Downside and concentration | D was multiplied by sqrt(h). HHI was rescaled to a clipped zero-to-one concentration score. | D is the root mean squared negative return with no extra horizon factor. C is mean raw HHI. | The penalty magnitudes now match Eqs. 23 and 26. |
| Score input to the loss | Each date's four utility scores were z-scored and clipped before target construction. | Raw Eq. 11 scores enter the temperature softmax. | Date-wise z-scoring changes the effective target temperature and is not in Eq. 12. |
| Training objective | The default used KL divergence to a target derived from z-scored scores, plus regret; a separate hard-label cross-entropy branch was disabled by its default weight. | The loss is explicit soft-target cross-entropy on raw-score targets plus regret weight 0.25. | KL and cross-entropy have the same model gradient for a fixed target, but the former implementation used a different target construction and exposed an unnecessary mixture. |
| Reference history | The committee reference used only the most recent min(756, n) training observations. | Eq. 14 is evaluated over the complete training history. | The reference now corresponds to the stated multi-year training-period selection. |
| Single-router decision | A stateful log-probability switching margin compared the current candidate with the previously selected expert. | Each date uses the stateless top-two probability difference; low confidence falls back to the training reference. | The old rule was path dependent and did not implement Eqs. 15–16. |
| Confidence grid | The single-router grid included 0.02 and 1.50; the committee grid included 1.50. | The grid is exactly 0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00. | Confidence thresholds now lie in the probability-difference range. |
| Committee aggregation | A routed expert was used only when every member agreed; any disagreement fell back to the reference. | The largest vote count wins; ties use ensemble-mean probabilities. | Unanimity is materially more conservative than the plurality rule in Eq. 17. |
| Reference blend | A convex reference/routed blend was already present. | The blend is retained and named directly as alpha in Eq. 18. | This scientifically required stability mechanism was preserved. |
| Training/test responsibility | A member-training command could also construct and evaluate the test split unless a special flag was supplied. | Member training always uses training and validation only. Only the committee command performs final chronological evaluation after calibration. | The command boundary now makes accidental member-level test evaluation harder. |
| Auditability | Caches and checkpoints were not bound to the package source and loaded-data identity. | Checkpoints and calibration records carry an implementation ID, a hash of package Python files, and a hash of the loaded features, returns, dates, tickers, and split boundaries; incompatible members are rejected. | Changes to the hashed source or loaded observations are detected. Raw NPZ hashes and environment versions still need separate archiving. |

Because these changes alter expert paths, supervision, optimization, routing, and aggregation, outputs from commit 1c62c3f are not interchangeable with outputs from the current implementation. New results must be generated and reported as a new execution.

The final portfolio simulator still uses the common equal-weight initialization at the first evaluation date. That backtest convention is separate from the continuous preceding-allocation convention used to construct Eq. 25 training and validation targets.

## Chronology and the meaning of “unseen”

The executable protocol is chronological:

- training observations end before 2023;
- validation covers 2023 and is used for checkpoint and deployment calibration;
- the final evaluation interval is 2024–2025;
- forward targets stay inside their training or validation split;
- the member-training command does not build or score test panels;
- the committee freezes training/validation choices before building the test panels, and each portfolio decision precedes the return it realizes.

These properties protect the computation from direct look-ahead. They do not make the 2024–2025 interval a pristine external test set. Those data were available during method development, so researcher exposure and iterative design choices cannot be undone by code cleanup. The defensible description is “chronological retrospective evaluation,” accompanied by this limitation, rather than “previously unseen data.”

The member-training command loads each complete NPZ container to validate it and compute a data fingerprint. It does not construct test windows or use test-period values in optimization or checkpoint selection. “Training does not evaluate test” is therefore a model-protocol statement, not a claim that test bytes are never read from storage.

## Removed material and preserved scientific components

The release omits:

- third-party baseline implementations;
- ablation-only scripts and outputs;
- report generators containing numbers that are not traceable to included run artifacts;
- generated checkpoints, caches, result tables, and datasets.

The release retains:

- all four primary portfolio experts;
- dynamic training targets;
- the training-only reference expert;
- confidence fallback and committee voting;
- alpha blending;
- chronological validation calibration and final evaluation.

The retained reference and chronology are necessary to execute the primary method. Removing them would simplify the repository by changing the scientific method.

## Evidence and claim boundaries

The source supports code-level claims about the implemented formulas, split ordering, output checks, and artifact fingerprints. It does not by itself support any numerical performance claim.

In particular, the current repository cannot establish:

- that a manuscript table was produced by this exact implementation;
- that the expected NPZ filenames identify an official or public download;
- that the supplied panels are byte-identical to those used for a prior run;
- that prior checkpoints used the current score, loss, or voting rules;
- that removed baselines or ablations can be regenerated;
- that the 2024–2025 interval was untouched during research development.

The absence of data and model artifacts means the paper metrics cannot presently be confirmed or rerun from this repository alone. A successful execution with newly supplied panels would be a rerun under the current implementation, not retroactive proof of the provenance of old numbers.

The reviewer's empirical questions also remain open. Validation-selected static expert mixtures, simpler causal selectors, and time-resolved attribution of gains to market episodes have not been run in this revision. Formula corrections alone do not demonstrate persistent value from adaptive routing, and terminal wealth cannot establish that conclusion. These comparisons require the exact market panels and fresh runs before the corresponding performance claims can be assessed.

## Minimum record for a new reproducible run

Archive the following together before making new numerical claims:

1. the Git commit and clean-tree status;
2. Python, dependency, OS, and accelerator versions;
3. all eight input-file hashes and their lawful provenance;
4. the saved implementation, source, dataset, and data fingerprints;
5. every member seed, configuration, checkpoint, training history, and runtime audit;
6. the full committee calibration grid and selected per-market reference, confidence threshold, and alpha;
7. final routes, portfolio weights, turnover, metrics, and logs;
8. an explicit disclosure that the evaluation is retrospective if the same interval informed method development.

Only results created from that archived bundle should be presented as outputs of the current paper-equation implementation.

## Code verification on 7 November 2026

Nine numerical checks of Eqs. 11–18 and B.3 passed. A separate three-member CPU run completed training, checkpoint loading, validation calibration, and final evaluation on an artificially generated panel. The checks covered portfolio feasibility, calibrated vote counts, rejection of incompatible checkpoints, and rejection of changed input observations. Additional checks covered chronological windows and horizons of one and five observations. The default model contains 819,568 parameters, matching Appendix B.2.

These engineering checks ran outside the release directory. Generated verification data and results are not manuscript experiments. The research datasets and original experiment artifacts remain unavailable, so this verification does not establish the paper's performance claims. Black, isort, and static undefined-name/import checks also passed.
