# Bias Detection and Mitigation — Prompt Firewall Dataset

## 1. What “bias” means here

This project does **not** have demographic fields (age, gender, location). Instead we analyze **categorical subgroups** that affect firewall behavior:

| Slice dimension | Why it matters |
|---|---|
| `source` | Different Hugging Face datasets use different labeling styles |
| `attack_type` | Jailbreak tactics (DAN, roleplay, system leak) vary in frequency |
| `prompt_length_bucket` | Short vs. long prompts may be detected differently |

Bias = systematic **representation skew** or **label-rate disparity** across slices that can cause over-refusal, under-detection, or poor generalization.

## 2. Detection — data slicing (Fairlearn)

**Tool:** [Fairlearn](https://fairlearn.org/) `MetricFrame` with `selection_rate` (injection rate per slice).

**Code:** `src/firewall/data/bias_report.py`

**Slices computed:**
- `by_source`
- `by_attack_type`
- `by_prompt_length_bucket` (short / medium / long / very_long / extreme)

**Per-slice metrics:**
- Row count and share of corpus
- Label counts (`INJECTION` / `BENIGN`)
- Injection rate / benign rate
- Fairlearn disparity (max − min injection rate across groups)

**Output:** `data/registry/v1.0/bias_report.json`

```json
{
  "before_mitigation": { "...": "raw combined pool" },
  "after_mitigation": { "...": "balanced training corpus" },
  "mitigation_applied": ["oversampled attack_type=...", "balanced_sample target=60000"],
  "remaining_warnings": []
}
```

**Warnings triggered when:**
- Slice has &lt; 100 rows (underrepresented)
- Slice is &lt; 5% of corpus
- Injection rate &lt; 15% or &gt; 85%
- Fairlearn disparity &gt; 25%

## 3. Model performance slicing (Fairlearn)

After evaluation, slice-wise **precision, recall, and false-positive rate** are computed per `attack_type`, `source`, and `prompt_length_bucket`.

**Code:** `src/firewall/data/model_bias.py` → called from `pipelines/evaluate_firewall.py`

**Output:** `reports/eval_<benchmark>.json` → `slice_performance` section

This answers: *“Does the firewall perform worse on certain attack types?”*

## 4. Mitigation strategies

| Strategy | When | Implementation |
|---|---|---|
| **Attack-type oversampling** | Rare tactics underrepresented | `mitigate_slice_representation()` in ingest |
| **Balanced label sampling** | INJECTION/BENIGN imbalance | `balanced_sample()` |
| **Stratified splits** | Preserve label ratio in train/val/test | `create_stratified_splits()` |
| **Held-out benchmarks** | Avoid eval contamination | Salad-Data in `benchmarks/` only |
| **Threshold tuning** (future) | High FPR on benign slice | Adjust policy thresholds on validation set |

Configured in `config/app.yaml`:

```yaml
bias:
  mitigate_attack_type_slices: true
  min_rows_per_attack_type: 500
```

## 5. Airflow integration

| Task | Role |
|---|---|
| `ingest_datasets` | Builds comparative bias report (before/after mitigation) |
| `verify_bias_report` | Confirms report exists; surfaces slice counts + Fairlearn disparity |
| `evaluate_firewall` | Adds model slice performance to eval JSON |

## 6. How to run locally

```bash
# Re-ingest and regenerate bias report
python pipelines/ingest_dataset.py --config config/app.yaml

# Inspect report
cat data/registry/v1.0/bias_report.json

# Run bias unit tests
pytest tests/unit/test_bias_report.py -q
```

## 7. Trade-offs

- Balancing to 50/50 labels **removes natural priors** — good for training stability, not for production base rates.
- Attack-type oversampling with replacement **duplicates prompts** — increases tactic coverage but may overfit templates.
- Single training source (WildJailbreak) in v1.0 limits cross-source bias analysis until more datasets are added.
