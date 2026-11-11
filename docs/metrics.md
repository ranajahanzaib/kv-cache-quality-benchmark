# Metrics

- **Accuracy:** correct scored answers divided by answers with a determinate correctness value.
- **Baseline-conditioned retention:** among cases answered correctly by baseline, the fraction also answered correctly by the candidate. Compute from paired fixture IDs; it is unavailable if the baseline has no correct cases.
- **Regression / improvement:** candidate changes a correct baseline to incorrect, or an incorrect baseline to correct. Persistent failures are separate.
- **Process RSS:** a point sample of the server process after a completion. RSS is not allocated KV memory and can include model weights and runtime overhead.
- **KV allocation:** unavailable unless explicitly emitted by the installed runtime. This implementation never substitutes an estimate.
- **Prompt length:** exact token count from `/tokenize`, distinct from configured context capacity.
