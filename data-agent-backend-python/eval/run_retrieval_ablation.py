from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repo root is importable when executed as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psycopg

from app.config import settings
from app.graph.nodes.table_relation import ColumnScore, TableRelationNode
from app.retrieval.retriever import VectorRetriever
from app.retrieval.schema_models import DbForeignKeySchemaView, DbTableSchemaView
from app.retrieval.schema_store import SchemaStore
from eval.score import extract_tables_from_sql

DEV_JSON_PATH = Path("/Users/johnnbb/Desktop/data agent/dev_20240627/dev.json")
TARGET_DBS = ("california_schools", "debit_card_specializing")
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)
RESULTS_DIR = REPO_ROOT / "eval" / "results"


class CachedVectorRetriever(VectorRetriever):
    """Caches query embeddings in-memory to avoid redundant API calls across identical query probes."""

    def __init__(self) -> None:
        super().__init__()
        self._embed_cache: dict[str, list[float]] = {}

    def embed_query(self, query: str) -> list[float]:
        if query not in self._embed_cache:
            self._embed_cache[query] = super().embed_query(query)
        return self._embed_cache[query]


def load_all_tables_for_db(schema_store: SchemaStore, database_id: str) -> dict[str, DbTableSchemaView]:
    sql = """
        SELECT
            t.id as table_id,
            t.name as table_name,
            t.description as table_desc,
            c.id as col_id,
            c.name as col_name,
            c.type as col_type,
            c.description as col_desc,
            c.is_primary_key as col_pk
        FROM db_table t
        LEFT JOIN db_column c ON t.id = c.table_id
        WHERE t.database_id = %(database_id)s
        ORDER BY t.name, c.name
    """
    tables = schema_store._fetch_tables_with_columns(sql, {"database_id": database_id})
    return {str(t.id): t for t in tables}


def compute_pruned_column_count(
    table_ids: list[str],
    tables_by_id: dict[str, DbTableSchemaView],
    high_sim_col_ids_by_table: dict[str, set[str]],
    fk_col_ids_by_table: dict[str, set[str]] | None = None,
) -> int:
    total = 0
    for tid in table_ids:
        table = tables_by_id.get(tid)
        if not table:
            continue
        selected = set(high_sim_col_ids_by_table.get(tid, set()))
        if fk_col_ids_by_table:
            selected |= fk_col_ids_by_table.get(tid, set())
        if selected:
            total += sum(1 for c in table.columns if str(c.id) in selected)
        else:
            total += len(table.columns)
    return total


def main() -> None:
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with DEV_JSON_PATH.open("r", encoding="utf-8") as f:
        all_dev = json.load(f)

    questions = [item for item in all_dev if item.get("db_id") in TARGET_DBS]
    print(f"Loaded {len(questions)} questions from {DEV_JSON_PATH} for {TARGET_DBS}")

    retriever = CachedVectorRetriever()
    schema_store = SchemaStore()

    tables_cache: dict[str, dict[str, DbTableSchemaView]] = {}
    fks_cache: dict[str, list[DbForeignKeySchemaView]] = {}
    for db_id in TARGET_DBS:
        tables_cache[db_id] = load_all_tables_for_db(schema_store, db_id)
        fks_cache[db_id] = schema_store.find_foreign_keys_by_database_id(db_id)
        print(f"  DB {db_id}: {len(tables_cache[db_id])} tables, {len(fks_cache[db_id])} foreign keys")

    # Reusable node instance (uses its select_threshold logic; no LLM calls made here)
    tr_node = TableRelationNode(schema_store=schema_store)

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(questions):
        q_id = item["question_id"]
        db_id = item["db_id"]
        q_text = item["question"]
        gold_sql = item["SQL"]
        difficulty = item.get("difficulty", "unknown")
        gold_tables = sorted(extract_tables_from_sql(gold_sql, dialect="sqlite"))
        gold_table_set = set(gold_tables)

        # 1. Leakage audit: search_questions without vs with exclude_question
        q_norm = q_text.strip().lower()
        q_before = retriever.search_questions(q_text, db_id, top_k=4)
        q_after = retriever.search_questions(q_text, db_id, top_k=4, exclude_question=q_text)

        top1_exact_before = bool(q_before and q_before[0].content.strip().lower() == q_norm)
        top4_exact_before = any(r.content.strip().lower() == q_norm for r in q_before)
        top1_exact_after = bool(q_after and q_after[0].content.strip().lower() == q_norm)
        top4_exact_after = any(r.content.strip().lower() == q_norm for r in q_after)
        top1_score_before = q_before[0].score if q_before else None
        top1_score_after = q_after[0].score if q_after else None

        # 2. Fetch tables (top 10) and columns (top 30) once per question
        table_docs = retriever.search_tables(q_text, db_id, top_k=10)
        column_docs = retriever.search_columns(q_text, db_id, top_k=30)

        table_scores: dict[str, float] = {}
        for r in table_docs:
            tid = r.metadata.get("tableId") or r.metadata.get("table_id")
            if tid:
                tid_str = str(tid)
                table_scores[tid_str] = max(table_scores.get(tid_str, 0.0), float(r.score))

        column_scores: list[ColumnScore] = []
        for r in column_docs:
            tid = r.metadata.get("tableId") or r.metadata.get("table_id")
            cid = r.metadata.get("columnId") or r.metadata.get("column_id")
            if tid and cid:
                column_scores.append(ColumnScore(table_id=str(tid), column_id=str(cid), score=float(r.score)))

        tables_by_id = tables_cache[db_id]
        foreign_keys = fks_cache[db_id]

        # 3. Threshold scan (0.20, 0.30, 0.40, 0.50, 0.60)
        threshold_results: dict[str, Any] = {}
        seed_040_names: set[str] = set()
        fk_040_names: set[str] = set()
        seed_040_all_recalled = False
        fk_040_all_recalled = False

        for thresh in THRESHOLDS:
            tr_node.base_high_similarity_threshold = thresh
            sel = tr_node.select_threshold(table_scores, column_scores)
            seed_ids = sel.table_ids
            seed_names = {tables_by_id[tid].name.lower() for tid in seed_ids if tid in tables_by_id}

            recall = (len(seed_names & gold_table_set) / len(gold_table_set)) if gold_table_set else 1.0
            all_recalled = gold_table_set.issubset(seed_names)
            pruned_cols = compute_pruned_column_count(
                seed_ids, tables_by_id, sel.high_similarity_column_ids_by_table
            )

            threshold_results[f"{thresh:.2f}"] = {
                "effective_threshold": round(sel.threshold, 4),
                "seed_table_names": sorted(seed_names),
                "seed_table_count": len(seed_ids),
                "pruned_column_count": pruned_cols,
                "gold_table_recall": round(recall, 6),
                "all_tables_recalled": all_recalled,
            }

            if abs(thresh - 0.40) < 1e-6:
                # 4. Foreign Key Expansion (Off vs On at threshold 0.40)
                related_fks = [
                    fk
                    for fk in foreign_keys
                    if str(fk.source_column.db_table.id) in seed_ids
                    or str(fk.target_column.db_table.id) in seed_ids
                ]
                fk_table_ids: set[str] = set()
                fk_cols_by_table: dict[str, set[str]] = {}
                for fk in related_fks:
                    s_tid = str(fk.source_column.db_table.id)
                    t_tid = str(fk.target_column.db_table.id)
                    fk_table_ids.add(s_tid)
                    fk_table_ids.add(t_tid)
                    fk_cols_by_table.setdefault(s_tid, set()).add(str(fk.source_column.id))
                    fk_cols_by_table.setdefault(t_tid, set()).add(str(fk.target_column.id))

                final_ids = list(dict.fromkeys([*seed_ids, *fk_table_ids]))
                fk_names = {tables_by_id[tid].name.lower() for tid in final_ids if tid in tables_by_id}
                fk_recall = (len(fk_names & gold_table_set) / len(gold_table_set)) if gold_table_set else 1.0
                fk_all_recalled = gold_table_set.issubset(fk_names)
                fk_pruned_cols = compute_pruned_column_count(
                    final_ids, tables_by_id, sel.high_similarity_column_ids_by_table, fk_cols_by_table
                )

                seed_040_names = seed_names
                fk_040_names = fk_names
                seed_040_all_recalled = all_recalled
                fk_040_all_recalled = fk_all_recalled
                fk_040_info = {
                    "seed_table_names": sorted(seed_040_names),
                    "final_table_names": sorted(fk_040_names),
                    "seed_table_count": len(seed_ids),
                    "final_table_count": len(final_ids),
                    "seed_pruned_columns": pruned_cols,
                    "final_pruned_columns": fk_pruned_cols,
                    "seed_gold_table_recall": round(recall, 6),
                    "final_gold_table_recall": round(fk_recall, 6),
                    "seed_all_tables_recalled": seed_040_all_recalled,
                    "final_all_tables_recalled": fk_040_all_recalled,
                    "rescued_by_fk": (not seed_040_all_recalled) and fk_040_all_recalled,
                }

        rec = {
            "question_id": q_id,
            "db_id": db_id,
            "difficulty": difficulty,
            "question": q_text,
            "gold_tables": gold_tables,
            "leakage_audit": {
                "top1_exact_before": top1_exact_before,
                "top4_exact_before": top4_exact_before,
                "top1_exact_after": top1_exact_after,
                "top4_exact_after": top4_exact_after,
                "top1_score_before": round(top1_score_before, 6) if top1_score_before is not None else None,
                "top1_score_after": round(top1_score_after, 6) if top1_score_after is not None else None,
            },
            "thresholds": threshold_results,
            "fk_ablation_040": fk_040_info,
        }
        records.append(rec)

        if (idx + 1) % 30 == 0 or (idx + 1) == len(questions):
            print(f"Processed {idx + 1}/{len(questions)} questions ({time.time() - t_start:.1f}s)")

    jsonl_path = RESULTS_DIR / "retrieval_ablation_153.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Aggregate summary
    n = len(records)
    leak_top1_before = sum(1 for r in records if r["leakage_audit"]["top1_exact_before"])
    leak_top4_before = sum(1 for r in records if r["leakage_audit"]["top4_exact_before"])
    leak_top1_after = sum(1 for r in records if r["leakage_audit"]["top1_exact_after"])
    leak_top4_after = sum(1 for r in records if r["leakage_audit"]["top4_exact_after"])
    scores_before = [r["leakage_audit"]["top1_score_before"] for r in records if r["leakage_audit"]["top1_score_before"] is not None]
    scores_after = [r["leakage_audit"]["top1_score_after"] for r in records if r["leakage_audit"]["top1_score_after"] is not None]

    threshold_summary: dict[str, Any] = {}
    for thresh in THRESHOLDS:
        key = f"{thresh:.2f}"
        threshold_summary[key] = {
            "gold_table_recall": round(sum(r["thresholds"][key]["gold_table_recall"] for r in records) / n, 4),
            "all_tables_recalled_rate": round(sum(1 for r in records if r["thresholds"][key]["all_tables_recalled"]) / n, 4),
            "all_tables_recalled_count": sum(1 for r in records if r["thresholds"][key]["all_tables_recalled"]),
            "avg_seed_tables": round(sum(r["thresholds"][key]["seed_table_count"] for r in records) / n, 4),
            "avg_pruned_columns": round(sum(r["thresholds"][key]["pruned_column_count"] for r in records) / n, 4),
        }

    fk_missed_before = sum(1 for r in records if not r["fk_ablation_040"]["seed_all_tables_recalled"])
    fk_rescued = sum(1 for r in records if r["fk_ablation_040"]["rescued_by_fk"])

    per_db_summary: dict[str, Any] = {}
    for db_id in TARGET_DBS:
        sub = [r for r in records if r["db_id"] == db_id]
        sn = len(sub)
        per_db_summary[db_id] = {
            "count": sn,
            "top1_exact_before": sum(1 for r in sub if r["leakage_audit"]["top1_exact_before"]),
            "top4_exact_before": sum(1 for r in sub if r["leakage_audit"]["top4_exact_before"]),
            "fk_off_040_all_tables_recalled_rate": round(sum(1 for r in sub if r["fk_ablation_040"]["seed_all_tables_recalled"]) / sn, 4),
            "fk_on_040_all_tables_recalled_rate": round(sum(1 for r in sub if r["fk_ablation_040"]["final_all_tables_recalled"]) / sn, 4),
            "fk_rescued_count": sum(1 for r in sub if r["fk_ablation_040"]["rescued_by_fk"]),
        }

    summary = {
        "total_questions": n,
        "databases": list(TARGET_DBS),
        "leakage_audit": {
            "top1_exact_before_count": leak_top1_before,
            "top1_exact_before_rate": round(leak_top1_before / n, 4),
            "top4_exact_before_count": leak_top4_before,
            "top4_exact_before_rate": round(leak_top4_before / n, 4),
            "top1_exact_after_count": leak_top1_after,
            "top4_exact_after_count": leak_top4_after,
            "avg_top1_score_before": round(sum(scores_before) / len(scores_before), 4),
            "avg_top1_score_after": round(sum(scores_after) / len(scores_after), 4),
        },
        "threshold_scan": threshold_summary,
        "fk_expansion_at_040": {
            "fk_off_gold_table_recall": round(sum(r["fk_ablation_040"]["seed_gold_table_recall"] for r in records) / n, 4),
            "fk_on_gold_table_recall": round(sum(r["fk_ablation_040"]["final_gold_table_recall"] for r in records) / n, 4),
            "fk_off_all_tables_recalled_rate": round(sum(1 for r in records if r["fk_ablation_040"]["seed_all_tables_recalled"]) / n, 4),
            "fk_on_all_tables_recalled_rate": round(sum(1 for r in records if r["fk_ablation_040"]["final_all_tables_recalled"]) / n, 4),
            "fk_off_all_tables_recalled_count": n - fk_missed_before,
            "fk_on_all_tables_recalled_count": sum(1 for r in records if r["fk_ablation_040"]["final_all_tables_recalled"]),
            "missed_before_fk_count": fk_missed_before,
            "rescued_by_fk_count": fk_rescued,
            "avg_tables_fk_off": round(sum(r["fk_ablation_040"]["seed_table_count"] for r in records) / n, 4),
            "avg_tables_fk_on": round(sum(r["fk_ablation_040"]["final_table_count"] for r in records) / n, 4),
            "avg_pruned_columns_fk_off": round(sum(r["fk_ablation_040"]["seed_pruned_columns"] for r in records) / n, 4),
            "avg_pruned_columns_fk_on": round(sum(r["fk_ablation_040"]["final_pruned_columns"] for r in records) / n, 4),
        },
        "per_db": per_db_summary,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }

    summary_path = RESULTS_DIR / "retrieval_ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Retrieval Ablation Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
