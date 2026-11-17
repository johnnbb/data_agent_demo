# Data Agent Text2SQL Evaluation Report

## 1. Evaluation Setup & Methodology

- **Target System**: `data-agent-backend-python` LangGraph Text2SQL pipeline (`EvidenceRecallNode -> SchemeRecallNode -> TableRelationNode -> FeasibilityAssessmentNode -> PlannerNode -> PlanExecuteNode <-> SqlGeneratorNode -> SqlExecuteNode`, skipping `HumanFeedbackNode` and `ReportGeneratorNode` / `PythonGeneratorNode` in offline SQL EX evaluation).
- **Benchmark Data**: BIRD `dev_20240627/dev.json` + read-only SQLite databases (`california_schools`, `debit_card_specializing`), PostgreSQL + pgvector `vector_store`.
- **Evaluation Splits**:
  - **Retrieval & Leakage Ablation**: All **153** dev questions (`89` `california_schools` + `64` `debit_card_specializing`).
  - **Baseline & Leakage A/B (`california_schools`, 30 Qs, seed=42)**: Stratified sample of `18 simple / 10 moderate / 2 challenging`.
  - **Full `california_schools` Evaluation**: All **89** dev questions (`54 simple / 30 moderate / 5 challenging`), plus exact 30-Q stratified slice extraction.
  - **Retail Transaction Holdout (`debit_card_specializing`, 15 Qs, seed=42)**: Stratified sample of `10 simple / 4 moderate / 1 challenging`.
- **Execution Accuracy (EX) Scorer (`eval/score.py`)**: Multiset row comparison (`compute_ex`) invariant to column aliases, column order, and row order, with 4-decimal float rounding, explicit empty-set flag, and `set_progress_handler` read-only SQLite timeout (`10.0s`). Unit-tested via `eval/test_score.py` (`4/4` passing).

---

## 2. Data Leakage Audit & Impact on E2E Accuracy

### 2.1 Question Bank Leakage Audit (153/153 Questions)
Sources: `eval/results/retrieval_ablation_summary.json`, `eval/results/retrieval_ablation_153.jsonl`.

| Condition | Top-1 Exact Question Match | Top-4 Exact Question Match | Avg Top-1 Cosine Score |
|---|---:|---:|---:|
| **Default (`leave_one_out=False`)** | **153 / 153 (100.00%)** | **153 / 153 (100.00%)** | **1.0000** |
| **Leave-One-Out (`exclude_question=q`)** | **0 / 153 (0.00%)** | **0 / 153 (0.00%)** | **0.7120** |

All 153 dev questions (`89` `california_schools` + `64` `debit_card_specializing`) are indexed in `vector_store` (`vectorType='questionKnowledge'`). Without leave-one-out filtering, every query retrieves its own ground-truth SQL example at cosine similarity `1.0000`.

### 2.2 E2E Leakage A/B & Pre-Optimization Baseline (`qwen3.8-max`)
Sources: `eval/results/e2e_eval_baseline_qwen38max.json`, `eval/results/e2e_group_a_leakage_30.jsonl`.

| Split | Group A (With Leakage) EX | Group B (Leave-One-Out Baseline) EX | Inflation Gap |
|---|---:|---:|---:|
| **`california_schools` (30 Qs)** | **23 / 30 (76.67%)** | **13 / 30 (43.33%)** | **-33.34 pp** |
| **`debit_card_specializing` Holdout (15 Qs)** | — | **7 / 15 (46.67%)** | — |

**Baseline Error Funnel (`qwen3.8-max`, `california_schools` 30 Qs LOO, 17 Failures)**:
- `1_feasibility_rejected`: `0`
- `2_table_recall_missed`: `2` (Mix Selector trimmed required table)
- `3_evidence_missed`: `3`
- `4_sql_exec_error_after_repairs`: `0`
- `5_sql_logic_mismatch`: **`12`** (`9/12` traced to unnecessary multi-step SQL+Python splitting and unrequested extra columns in `planner.st`, plus missing enum value examples and narrow `glossary` `top_k=4`).

---

## 3. Schema Recall Threshold & Foreign-Key Expansion Ablation (153 Questions)

Sources: `eval/results/retrieval_ablation_summary.json`, `eval/results/retrieval_ablation_153.jsonl`.

### 3.1 Dynamic Threshold Scan (FK Expansion Off)

| Base Threshold | Gold Table Recall (Macro) | All-Tables Recalled Rate | All-Tables Count | Avg Seed Tables | Avg Pruned Columns |
|---:|---:|---:|---:|---:|---:|
| `0.20` | 98.80% | 97.39% | 149 / 153 | 3.41 | 21.28 |
| `0.30` | 97.60% | 94.77% | 145 / 153 | 3.14 | 18.51 |
| **`0.40` (Prod)** | **89.32%** | **81.05%** | **124 / 153** | **2.45** | **8.29** |
| `0.50` | 75.05% | 54.25% | 83 / 153 | 1.64 | 14.29 |
| `0.60` | 70.81% | 46.41% | 71 / 153 | 1.46 | 23.39 |

- `0.40` achieves the minimum average column footprint (**8.29** columns/query vs `18.51–21.28` at lower thresholds; at `0.50–0.60`, fallback retention on tables with zero high-similarity columns inflates column counts back to `14.29–23.39`).

### 3.2 Foreign-Key Topology Expansion at Threshold `0.40`

| Setting | Gold Table Recall | All-Tables Recalled Rate | All-Tables Count | Avg Tables | Avg Pruned Columns |
|---|---:|---:|---:|---:|---:|
| **FK Expansion Off** | 89.32% | 81.05% | 124 / 153 | 2.45 | 8.29 |
| **FK Expansion On** | **97.49%** | **94.12%** | **144 / 153** | **2.76** | **8.71** |

- **Missed before FK**: `29` questions; **Rescued by FK expansion**: **`20 / 29` (68.97%)** at a cost of only `+0.31` tables and `+0.41` columns per query.
- **Per-Database Breakdown**:
  - `california_schools` (89 Qs): All-tables recall **`86.52% -> 100.00%`** (**12/12** missed rescued).
  - `debit_card_specializing` (64 Qs): All-tables recall **`73.44% -> 85.94%`** (**8** rescued).

---

## 4. Full-Dataset Evaluation After All 4 Optimizations (`glm-5` 89-Q + `glm-5.1` 15-Q Holdout)

Four orthogonal data/retrieval and multi-agent contract optimizations were landed and verified:
1. **Low-Cardinality Enum Value Sampling** (`app/retrieval/schema_models.py`, `app/graph/nodes/table_relation.py`): read-only SQLite sampling (`mode=ro`, `timeout=2.0s`, `LIMIT 16`) populates `Examples: [...]` on `Schema` (`examplesByTableCol`) for low-cardinality categorical/text columns (`<= 15` distinct values -> up to 5 samples; `> 15` -> 2 short samples).
2. **Expanded Glossary Recall** (`app/graph/nodes/evidence_recall.py`): increased `search_glossary(..., top_k=8)` from `4` to `8`.
3. **Planner Single-SQL-First & Exact Projection** (`app/prompts/planner.st`): eliminated unnecessary multi-step SQL+Python splits and unrequested projection columns.
4. **Contract Alignment & Casing Grounding** (`app/prompts/new-sql-generate.st`) + **SQLite Execution Timeout Guard** (`app/graph/nodes/sql_execute.py`): aligned `SELECT` projection against the original user question, enforced strict business-formula/column-set usage, mandated exact literal casing from `Examples: [...]`, and added a `10.0s` `set_progress_handler` read-only timeout to `SqlExecuteNode`.

Sources: `eval/results/e2e_eval_glm5_summary.json`, `eval/results/e2e_glm5_full_cs_89.jsonl`, `eval/results/e2e_glm51_holdout_15.jsonl`.

### 4.1 Quantitative Scorecard (`leave_one_out=True`)

| Metric | Pre-Opt Baseline (`qwen3.8-max`, 30-Q / 15-Q) | Post-Opt Full `california_schools` (`89` Qs, `glm-5`) | Post-Opt 30-Q Slice (`glm-5`) | Post-Opt Retail Holdout (`15` Qs, `glm-5.1`) |
|---|---:|---:|---:|---:|
| **Overall EX** | `43.33%` (CS) / `46.67%` (Retail) | **46 / 89 (51.69%)** | `13 / 30 (43.33%)` | **9 / 15 (60.00%)** |
| **Simple EX** | `50.00%` / `50.00%` | `30 / 54 (55.56%)` | `9 / 18 (50.00%)` | `7 / 10 (70.00%)` |
| **Moderate EX** | `30.00%` / `50.00%` | `15 / 30 (50.00%)` | `4 / 10 (40.00%)` | `2 / 4 (50.00%)` |
| **Challenging EX** | `50.00%` / `0.00%` | `1 / 5 (20.00%)` | `0 / 2 (0.00%)` | `0 / 1 (0.00%)` |
| **Feasibility False Rejects** | `0.00%` / `0.00%` | `3 / 89 (3.37%)` | `2 / 30 (6.67%)` | `0 / 15 (0.00%)` |
| **Pre-Mix -> Post-Mix Table Recall** | `100.00% -> 93.33%` (CS) | `89/89 (100.00%) -> 82/89 (92.13%)` | `30/30 -> 28/30 (93.33%)` | `12/15 (80.00%) -> 13/15 (86.67%)` |
| **Avg Tables (Pre -> Post Mix)** | `3.00 -> 1.77` | `2.98 -> 1.73` | `3.00 -> 1.73` | `2.80 -> 1.53` |
| **SQL 1st-Attempt -> Final Exec OK** | `100.00% -> 100.00%` | `84/86 (97.67%) -> 86/86 (100.00%)` | `28/28 -> 28/28 (100.00%)` | `15/15 -> 15/15 (100.00%)` |
| **SQL Self-Repair Rescues** | `0` triggered | `2` exec rescued, `1` EX rescued | `0` triggered | `0` triggered |
| **Node Errors** | `0` | `0` | `0` | `0` |

### 4.2 5-Stage Error Attribution Funnel

| Stage | `glm-5` Full (`89` Qs) | `glm-5` Slice (`30` Qs) | `glm-5.1` Retail Holdout (`15` Qs) |
|---|---:|---:|---:|
| `passed` (EX Match) | **46** | **13** | **9** |
| `1_feasibility_rejected` | 3 | 2 | 0 |
| `2_table_recall_missed` | 5 | 2 | 1 |
| `3_evidence_missed` | 6 | 5 | 1 |
| `4_sql_exec_error_after_repairs` | 0 | 0 | 0 |
| `5_sql_logic_mismatch` | 29 | 8 | 4 |
| **Total** | **89** | **30** | **15** |

---

## 5. Residual Failure Analysis & Next-Step Roadmap

### 5.1 Retail Holdout (`debit_card_specializing`, `6` Failed Questions)
1. **Benchmark Label Noise (Q1496)**: Question asks *"Which client segment consumed the least in September 2013?"* Our agent generated `GROUP BY T1."Segment"`, while BIRD Gold SQL grouped by `T1.CustomerID` (`GROUP BY T1.CustomerID`), ranking individual customers instead of segments.
2. **Yes/No Compound Question Column Granularity (Q1486)**: Computed exact numeric difference (`CZK - EUR`), but returned `(CZK_total, EUR_total, Difference)` instead of Gold SQL's single `Difference` column.
3. **Metric Definition Ambiguity (Q1473, Q1498, Q1481)**: `SUM(Consumption) / 12` vs `AVG(Consumption) / 12` for "average monthly consumption", `GROUP BY Date` vs `SUBSTR(Date, 5, 2)`, and multi-segment nested extremum logic (Q1481).
4. **Mix Selector Cross-Table Extremum Drop (Q1531)**: Mix Selector dropped summary table `yearmonth`, causing the top customer lookup to sum `transactions_1k` instead.

### 5.2 `california_schools` Full Run (`43` Failed Questions)
1. **Feasibility False Rejections (`3` Qs: Q1, Q67, Q79)**: E.g., Q1's `'Continuation School'` sits at index 6 of 12 distinct values and was truncated by the `rows[:5]` cap.
2. **Mix Selector Over-Trimming (`5` Qs: Q2, Q4, Q18, Q71, Q87)**: Post-FK recall was `89/89 (100%)`, but Mix Selector dropped `frpm` on questions containing near-synonym columns in `schools` (e.g., `Charter` vs `Charter School (Y/N)`).
3. **Unretrieved Implicit Evidence (`6` Qs: Q17, Q25, Q26, Q27, Q45, Q76)**: E.g., `"administrators manage school"` -> `AdmFName1` or `"Valid charter number"` -> `CharterNum IS NOT NULL` failed to rank within Top-8 dense vector results.
4. **SQL Logic & Projection Mismatches (`29` Qs)**:
   - **`18 / 29` matched exact row counts** with Gold SQL, failing due to extra context columns (`Q14`, `Q30`), concatenated name columns (`Q36` `AdmFName1 || ' ' || AdmLName1` vs separate columns), or code/name swaps (`Q28` `DOCType` vs `DOC`).
   - **`4 / 29` returned 0 rows** due to high-cardinality entity column confusion (`Q60`: `'San Joaquin'` matched to `City` instead of `County`; `Q86`: leading-zero code `'0040'` stripped to `'40'`).

### 5.3 Next-Step Optimization Directions
1. **Full Low-Cardinality Coverage + High-Cardinality Entity Value Index**: Raise low-cardinality (`<= 15`) sampling cap from `5` to `15` (fixes Q1), and build a lightweight inverted/BM25 index over entity values so literals like `'San Joaquin'` and `'0040'` bind to exact `(table, column, literal)` triples (fixes Q60, Q86).
2. **Mix Selector Bridge & Synonym Guard**: Preserve FK-linked tables when candidate count `<= 4` or tables share overlapping business columns (`Charter`/`FundingType`), recovering the `100%` post-FK recall (fixes 5 `california_schools` misses + Q1531).
3. **SQLGlot Post-Generation Output Schema Verifier**: Strip unrequested auxiliary columns / redundant aggregations in single-entity queries via AST inspection before execution (fixes Q14, Q30, Q1486).
4. **Dense + BM25 Hybrid Glossary Retrieval**: Pair pgvector cosine similarity with BM25 lexical matching to catch short field-mapping keywords missed by embeddings (fixes the 6 `3_evidence_missed` cases).

---

## 6. Regression & Security Verification

Executed after all code and prompt updates:
```bash
.venv/bin/pytest tests/test_plan_execute_and_human_feedback.py tests/test_sql_generate_and_execute.py tests/test_table_relation.py tests/test_evidence_recall.py eval/test_score.py -k "not full_pipeline and not graph and not node_full_pipeline" -v
```
- **Result**: **`24 passed, 4 deselected`** (independently verified by `verifier`), confirming zero regressions from leave-one-out hooks, low-cardinality sampling, glossary `top_k=8`, SQL timeout guard, and `100%` AST write-operation interception (`test_validate_sql_ast_security_interception`).
