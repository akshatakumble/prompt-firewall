# Bias Detection and Mitigation — Prompt Firewall Dataset

## 1. Definition of bias in this project

For the firewall training corpus, **bias** means systematic imbalance in label distribution or representation across dataset sources and attack categories. A biased corpus can cause:

- **Over-refusal**: blocking benign prompts from underrepresented sources
- **Under-detection**: missing attacks from rare attack types
- **Benchmark contamination**: training on held-out evaluation data

## 2. Detection via data slicing

We slice the unified schema on two categorical dimensions:

| Slice dimension | Rationale |
|---|---|
| `source` | Each Hugging Face dataset may have different label priors |
| `attack_type` | Jailbreak tactics (DAN, roleplay, system leak) vary in frequency |

Implementation: `src/firewall/data/bias_report.py` → `build_bias_report()`

Output: `data/registry/v1.0/bias_report.json`

Example metrics per slice:

- Row count
- Label counts (`INJECTION` / `BENIGN`)
- Injection rate

**Warning threshold**: injection rate below 15% or above 85% within a source slice triggers a warning.

## 3. Findings (v1.0)

On the WildJailbreak training sample (60K balanced rows before split):

- **Overall**: 50% INJECTION / 50% BENIGN (by design via `balanced_sample`)
- **By source**: single training source (`train_wildjailbreak`) at 50/50
- **By attack_type**: `benign` slice is 100% BENIGN; adversarial slices are 100% INJECTION — expected given schema mapping

No cross-source skew warnings were raised for v1.0 because balancing runs before stratified splitting.

## 4. Mitigation strategies applied

| Strategy | Implementation |
|---|---|
| **Balanced sampling** | `balanced_sample()` caps each label before combining sources |
| **Stratified splits** | 70/15/15 train/val/test stratified by `label` |
| **Held-out benchmark** | Salad-Data kept in `benchmarks/` only, never merged into training |
| **Duplicate removal** | `clean_dataframe()` drops empty and exact-duplicate prompts |
| **Validation gating** | GE runner hard-fails on null prompts, unknown labels; soft-warns on imbalance |

## 5. Trade-offs

- **Balancing reduces raw volume** from ~262K WildJailbreak rows to 60K — improves class balance but discards data diversity.
- **Single training source in v1.0** — faster to ship; additional sources (JailBreakV, Aurora-M, Simsonsun) planned for v1.1.
- **Attack-type slices are label-deterministic** — warnings focus on `source`-level skew, not tactic-level rarity.

## 6. Airflow integration

The DAG task `verify_bias_report` confirms `bias_report.json` and `manifest.json` exist after ingest and surfaces slice counts in the success email.

## 7. Future work

- Add per-`attack_type` minimum count thresholds before training
- Expand slicing to prompt length buckets (short vs. long prompts)
- Re-sample underrepresented tactics when multi-source ingest lands in v1.1
