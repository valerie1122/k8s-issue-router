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
