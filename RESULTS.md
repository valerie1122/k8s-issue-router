# Results (test = issues created 2026-07-01 .. 2026-09-22 with a 30-day answer, n=441)

Thresholds are picked on val (2026-04..06) and applied unchanged to test.
"Auto-route" = share of test issues the router assigns without a human; precision = share of those that were right.

| Router | top-1 | aim 90%: auto-route / precision | aim 95%: auto-route / precision | ECE | cost / 1k issues |
|---|---|---|---|---|---|
| majority (sig/node) | 0.40 | 0% | 0% | – | $0 |
| B0 TF-IDF + logreg | 0.75 | 61.5% / 0.897 | 49.7% / 0.936 | 0.32 | ~$0 |
| B1 bge-base kNN vote | 0.81 | 72.3% / 0.922 | 49.7% / 0.963 | 0.21 | ~$0 |
| B1b bge-base + logreg | 0.80 | 60.1% / 0.913 | 46.5% / 0.941 | 0.45 | ~$0 |
| M2 ModernBERT-base (calibrated, T=1.12) | 0.85 | 85.9% / 0.900 | 63.9% / **0.961** | **0.04** | ~$0 (CPU-servable) |
| B2 Haiku 5.5 zero-shot | **0.88** | 95.9% / 0.891 | 77.3% / 0.938 | 0.10 | $0.22 (batch) |
| Cascade: M2 -> Haiku, route on M2-Haiku agreement | – | 93.0% / 0.885 | **81.9%** / 0.939 | – | LLM on ~51% |

30 of 1098 first-pass batch requests failed and were re-run with a larger output limit (0 failures on retry). Any missing answer counts as "sent to a human".

## Findings
- A free model gets far: ModernBERT auto-routes 64% at 96% precision and is well calibrated (ECE 0.04).
- Haiku zero-shot is the most accurate single model (0.90 top-1) but its stated confidence is coarse (0.6 / 0.8 / 0.93), so it cannot hit a 95% bar by itself.
- Agreement is the useful LLM signal: M2 and Haiku agree on 83% of issues; when they agree M2 is right 94.6%, when they disagree 45.8%.
- Val -> test drift costs ~1-1.5 points of precision at the same threshold; report it, don't hide it.
- Haiku almost never says "no SIG" (6/65 on test issues that never got one) -> needs a separate rule or the agent layer.
- Caveat: test issues may predate the LLM's training cutoff (possible contamination). Shadow mode on brand-new issues is the clean check.

M2 training: 380 s on one L4, ~$0.10.

## Contamination check
Haiku 5.5 reliable-knowledge cutoff is June 2026; test issues start July 2026.
Haiku top-1: val (Apr-Jun, before cutoff) 0.874 vs test (after cutoff) 0.864. M2 (cannot be contaminated): 0.831 vs 0.853.
No sign that Haiku benefits from having seen the issues.

## Cost
Batch run of 1098 requests: $0.24 total ($0.22 per 1k issues). Real-time Haiku in the shadow service: ~$0.23 per 1k issues.

## Seed variance (ModernBERT-base, 3 seeds, same data and settings)
| seed | top-1 | aim 95%: auto-route / precision |
|---|---|---|
| 0 | 0.857 | 55.3% / 0.980 |
| 1 | 0.862 | 55.6% / 0.967 |
| 2 | 0.878 | 54.6% / 0.971 |
| first run (seed 0, before saving weights) | 0.853 | 63.9% / 0.961 |

top-1 0.866 ± 0.011. Coverage at the 95% bar moves 55-64% between runs: the val set (356 labeled issues) is small, so the
threshold it picks is noisy. Quote coverage as a range, not a single number.

Cascade with the served model (seed 0, agreement rule, thresholds from val): aim 95% -> test auto-routes 68.0%, precision 0.973 [0.95,0.99].

## Live shadow mode (started 2026-10-23 22:01 UTC)
First 13 h: 2 new issues (weekend night), both captured; 1 auto-routed, correct. Too small to mean anything yet.
Live LLM cost was $0.72 per 1k issues, ~3x the batch estimate: issues arrive ~1 per hour, so the 5-minute prompt cache is
never reused and every call pays the 1.25x cache-write price. Config v2 turns caching off for live traffic.

## Replay backtest (shadow decision code, test period in creation order)
`python replay.py` runs `shadow.decide` with `shadow.CONFIG` (v2) over the stored model outputs.
- 441 test issues with a 30-day answer: auto-routed 68.0%, precision 0.973 [0.95,0.99] (matches the offline cascade).
- Issues a human other than the author had to route: router would auto-route 31 of 47, 30 correct; humans took median 12 h (p75 91 h).
- **Weak spot: 33 of 67 issues that never got a SIG were auto-routed anyway.** Counting those as errors, strict precision is 0.877.
  A "should this be routed at all" classifier (bge + logreg) only reaches AUC 0.73; gating on it trades ~5 points of coverage for ~1 point of strict precision.
  From the 6-issue audit, these are duplicates, support questions, and bugs nobody picked up (where routing would arguably help).
  -> The agent layer targets exactly this: duplicate search + support detection.
- LLM spend under the $1/day budget rule: $0.49 over 109 days.

## Agent layer (M1: Haiku 5.5 + 3 tools, only on issues the cascade sends to a human)
317 human-bound issues (val+test), 0 errors, avg 2.8 turns, $0.0012 per issue ($0.39 total), p50 6.7 s.
Trust threshold 0.71 picked on val.

| | val | test |
|---|---|---|
| auto-route share of labeled issues | 78.7% | **77.8%** (was 68.0% without the agent) |
| precision on those | 0.950 | **0.959** [0.93,0.98] |
| strict precision (no-SIG auto-routes counted wrong) | 0.847 | 0.855 |

- Agent top-1 on these hard issues: 0.783. When it sides with ModernBERT it is right 87%, with Haiku 73%, with neither 58%:
  its main value is deciding which of two disagreeing models to trust.
- Says "none" rarely but accurately: 20 times, 17 truly no-SIG (85%), catching 25% of them. Never flagged a duplicate.
- Tool use: search_similar_issues 506, lookup_owners 381, get_sig_charter 109 calls.
- Shadow service config v3 adds this layer live.

## Error analysis (28 wrong auto-routes, val+test, cascade v3)
- ~1/3 CI-failure / flaky-test issues (`[Failing Test] ci-kubernetes-...`, "golang tip broken" x3 -> humans say architecture).
- ~12 at SIG boundaries where both answers are defensible: DRA (node vs scheduling), Windows (windows vs storage/node),
  kubelet networking (node vs network).
- The rest: cross-cutting umbrellas (PQC, release chores) and label oddities (a title starting "sig/network:" labeled api-machinery).

Tested and rejected: a CI-job -> SIG lookup built from test-infra (5,497 jobs). For issues naming a known job, the
job's SIG matched the human label only 45% of the time: generic jobs (pull-kubernetes-e2e-gce, ci-kubernetes-unit)
run every SIG's tests. The `[sig-x]` tag in the failing test's name matches 93% (104/112), and the models already see it.

## What "no SIG after 30 days" really is (131 val+test issues)
still open, never routed 49 (37%) · closed, mostly fixed via PR without a label 41 (31%) · stale/rotten 30 (23%) ·
support / wrong place 8 (6%) · duplicates 3 (2%).
So most auto-routes of these issues are not harmful; issues that truly should not be routed are ~2-3% of all issues.

## Day 4: bigger model, agent prompt v2, too-short rule (test set)
| | top-1 | aim 95%: auto-route / precision | real-time cost per 1k issues |
|---|---|---|---|
| Haiku 5.5 zero-shot | 0.880 | 77.3% / 0.938 | ~$0.7 |
| Sonnet 5.5 zero-shot | **0.907** (macro recall 0.916 vs 0.818) | **80.0% / 0.955** | ~$15 (7k-token prompt, no cache hits at 1 issue/h) |
| Cascade + agent v1 (served, config v4) | – | 76.0% / 0.961 [0.93,0.98]; strict 0.863 | ~$0.9 |
| Cascade + agent v2 prompt | – | 70.7% / 0.974 [0.95,0.99]; strict 0.879 | ~$0.9 |

- Sonnet alone roughly matches the whole cascade, at ~17x the live cost. The cascade's point is cost, not accuracy.
- Where the cascade spends (test): ModernBERT alone settles 19% of issues, 81% get a Haiku call ($0.58/1k), 26% also get the agent ($0.32/1k): ~$0.90 per 1k issues.
- Prompt v2 (explicit do-not-route cases) trades 5 points of coverage for 1.3 points of precision; "none" recall 25% -> 32%.
  Serving stays on v1 because the goal is max coverage at >= 95% precision.
- Too-short rule (< 30 body words -> human): 4% of issues; agent threshold re-tuned 0.71 -> 0.73.
- Neither LLM says "none" much: Sonnet 14/130, Haiku 13/130 on issues that never got a SIG.

## PyTorch portability check (2026-10-24): the job is already done there, by a Claude bot
Pulled 1 year of pytorch/pytorch (7,922 issues) to port the router. What the data showed:
- PyTorch routes with `oncall: <queue>` plus ~150 `module:` labels. 51% of issues get an oncall queue (69% of those are `pt2`).
- In **Feb 2026 maintainer routing collapsed from ~58% of issues to 1-8%**: github-actions took over (73-84% of first
  routing labels). The repo runs `.github/workflows/claude-issue-triage.yml` (plus a distributed-triage variant):
  a Claude-based issue triager in production.
- How often do humans correct it? On 3,219 bot-routed issues (Feb 15 - Sep 24, 30-day window):
  any routing-label change by a human 12.1%; a bot label removed 5.3%; a label added 9.6%;
  **oncall queue removed 3.3%** (48 of 1,471). `pt2` 13/1031, `distributed` 4/259.
  Caveat: "not corrected" is not "correct" (silence bias), so ~97% is an upper bound on its oncall precision.
- Consequence: after Feb 2026 almost every pytorch label is the bot's own output, so a model trained or scored on it
  would learn to copy the bot. Building the pytorch router was stopped here; the multi-repo code (router/repos.py) stays.

Takeaway for the k8s work: an LLM-first triager is what a large project chose to run in production; the k8s cascade
reaches similar precision on its auto-routed share (0.96 vs <= 0.97 upper bound) at about $0.9 per 1k issues.

## Leak ablation (`leak_ablation.py`, TF-IDF + logreg, same split and labels)
| input text | test top-1 | top-1 where author did not route | aim 95%: auto-route / precision |
|---|---|---|---|
| clean: creation-time text, commands stripped (used everywhere) | 0.751 | 0.756 | 49.7% / 0.936 |
| post-edit title/body, commands stripped | 0.739 | 0.728 | 49.9% / 0.927 |
| post-edit text, `/sig` etc. left in | **0.819** | 0.744 | **59.6%** / 0.954 |

- Leaving routing commands in inflates top-1 by ~7 points and coverage by ~10, all of it on issues where the author
  named the SIG: the model copies the answer.
- Restoring pre-edit titles/bodies made no measurable difference for this model. Kept anyway (it is the correct input).

## Live shadow mode, first 6 days (2026-10-23 22:01 UTC to 2026-10-29)
33 new issues seen; 30 already have a human SIG label (provisional: some are not 30 days old yet).
- Auto-routed 25 of 30 (83%), **all 25 correct** (Wilson 95% CI 0.87-1.00). ModernBERT top-1 86.7%, Haiku top-1 90.0%.
- Latency p50 3.3 s, p95 13.3 s (agent cases). LLM cost $0.021 total, $0.65 per 1k issues; agent ran 8 times, 0 errors.
- Rows by config: v1 2, v3 4, v4 27 (report aggregates all three).
Consistent with the replay (76% at 0.96); the sample is small, so the CI is what to quote.

## Self-review experiment (`b2_review.py`): Haiku re-checks its own first answer
Haiku sees the issue plus its first answer and reason, and keeps or changes it. 992 of 1,098 reviews returned (106 errors).
On the 391 labeled test issues answered in both passes:
- top-1 0.903 first pass -> 0.900 after review. Changed 29 answers: 9 wrong->right, 10 right->wrong. **No net gain.**
- But *whether it changed its mind* is a signal: accuracy 0.923 when the review keeps the answer vs 0.621 when it changes it.
- Cascade layers 1-2 (ModernBERT agreement gate): 69.8% / 0.978 with first-pass answers vs 70.6% / 0.978 with reviewed ones,
  within noise. The gate already requires agreement with an independent model, so a same-model re-check adds little.

## Answer stability (`stability.sh`): Haiku run 3 times on the same prompt
422 labeled test issues answered in all 3 runs.
- top-1 per run: 0.903 / 0.915 / 0.908.
- Same answer in all 3 runs on **96.0%** of issues (accuracy 0.919); on the 4% where runs disagree (n=17), run 1 is right 52.9%.
- Cascade layers 1-2: the decision changes between runs on 5 of 422 issues (1.2%); auto 69.9-70.1%, precision 0.973 in every run.
- Self-consistency (route only if all 3 runs agree with ModernBERT): 70.1% / 0.973, no gain for 3x the Haiku cost.
LLM sampling noise is real but small and sits on the hard cases; the offline cascade numbers are stable to it.
