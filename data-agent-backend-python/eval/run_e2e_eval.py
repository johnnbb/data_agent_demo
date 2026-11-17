from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings
from app.graph.edges import feasibility_assessment_edge, sql_executor_edge
from app.graph.nodes.evidence_recall import EvidenceRecallNode
from app.graph.nodes.feasibility_assessment import FeasibilityAssessmentNode
from app.graph.nodes.plan_execute import PlanExecuteNode
from app.graph.nodes.planner import PlannerNode
from app.graph.nodes.scheme_recall import SchemeRecallNode
from app.graph.nodes.sql_execute import SqlExecuteNode, locate_sqlite_database
from app.graph.nodes.sql_generator import SqlGeneratorNode
from app.graph.nodes.table_relation import ColumnScore, TableRelationNode
from app.graph.spec import NodeName, StateKey
from app.graph.state import AgentState
from app.retrieval.retriever import VectorRetriever
from app.retrieval.schema_store import SchemaStore
from eval.score import compute_ex, execute_sqlite_readonly, extract_tables_from_sql

DEV_JSON_PATH = Path("/Users/johnnbb/Desktop/data agent/dev_20240627/dev.json")
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def stratified_sample(
    questions: list[dict[str, Any]],
    targets: dict[str, int],
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for diff, count in targets.items():
        pool = sorted(
            [q for q in questions if q.get("difficulty") == diff],
            key=lambda x: x["question_id"],
        )
        if len(pool) < count:
            raise ValueError(f"Not enough '{diff}' questions: need {count}, have {len(pool)}")
        sampled.extend(rng.sample(pool, count))
    sampled.sort(key=lambda x: x["question_id"])
    return sampled


def load_eval_splits() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with DEV_JSON_PATH.open("r", encoding="utf-8") as f:
        all_dev = json.load(f)

    cs_all = [q for q in all_dev if q.get("db_id") == "california_schools"]
    dc_all = [q for q in all_dev if q.get("db_id") == "debit_card_specializing"]

    # Deterministic 30-Q stratified sample: 18 simple, 10 moderate, 2 challenging
    cs_sample = stratified_sample(cs_all, {"simple": 18, "moderate": 10, "challenging": 2}, seed=42)
    # Deterministic 15-Q stratified holdout: 10 simple, 4 moderate, 1 challenging
    dc_sample = stratified_sample(dc_all, {"simple": 10, "moderate": 4, "challenging": 1}, seed=42)
    return cs_sample, dc_sample


def build_eval_graph(chat_model: str):
    """Build lightweight eval graph reusing exact production node classes, skipping HumanFeedback and Report/Python."""
    retriever = VectorRetriever()
    schema_store = SchemaStore()

    evidence_node = EvidenceRecallNode(retriever=retriever, chat_model=chat_model)
    scheme_node = SchemeRecallNode(retriever=retriever)
    table_relation_node = TableRelationNode(schema_store=schema_store, chat_model=chat_model)
    feasibility_node = FeasibilityAssessmentNode(chat_model=chat_model)
    planner_node = PlannerNode(chat_model=chat_model)
    plan_execute_node = PlanExecuteNode()
    sql_gen_node = SqlGeneratorNode(chat_model=chat_model)
    sql_exec_node = SqlExecuteNode()

    for n in (evidence_node, table_relation_node, feasibility_node, planner_node, sql_gen_node):
        n.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            timeout=45.0,
            max_retries=2,
        )

    def eval_plan_executor_edge(state: AgentState) -> str:
        next_node = state.get(StateKey.NEXT_NODE.value, END)
        if next_node == NodeName.SQL_GENERATION.value:
            return NodeName.SQL_GENERATION.value
        return END

    builder = StateGraph(AgentState)
    builder.add_node(NodeName.EVIDENCE_RECALL.value, evidence_node)
    builder.add_node(NodeName.SCHEMA_RECALL.value, scheme_node)
    builder.add_node(NodeName.TABLE_RELATION.value, table_relation_node)
    builder.add_node(NodeName.FEASIBILITY_ASSESSMENT.value, feasibility_node)
    builder.add_node(NodeName.PLANNER.value, planner_node)
    builder.add_node(NodeName.PLAN_EXECUTION.value, plan_execute_node)
    builder.add_node(NodeName.SQL_GENERATION.value, sql_gen_node)
    builder.add_node(NodeName.SQL_EXECUTION.value, sql_exec_node)

    builder.add_edge(START, NodeName.EVIDENCE_RECALL.value)
    builder.add_edge(NodeName.EVIDENCE_RECALL.value, NodeName.SCHEMA_RECALL.value)
    builder.add_edge(NodeName.SCHEMA_RECALL.value, NodeName.TABLE_RELATION.value)
    builder.add_edge(NodeName.TABLE_RELATION.value, NodeName.FEASIBILITY_ASSESSMENT.value)
    builder.add_conditional_edges(
        NodeName.FEASIBILITY_ASSESSMENT.value,
        feasibility_assessment_edge,
        {
            NodeName.PLANNER.value: NodeName.PLANNER.value,
            END: END,
        },
    )
    builder.add_edge(NodeName.PLANNER.value, NodeName.PLAN_EXECUTION.value)
    builder.add_conditional_edges(
        NodeName.PLAN_EXECUTION.value,
        eval_plan_executor_edge,
        {
            NodeName.SQL_GENERATION.value: NodeName.SQL_GENERATION.value,
            END: END,
        },
    )
    builder.add_edge(NodeName.SQL_GENERATION.value, NodeName.SQL_EXECUTION.value)
    builder.add_conditional_edges(
        NodeName.SQL_EXECUTION.value,
        sql_executor_edge,
        {
            NodeName.SQL_GENERATION.value: NodeName.SQL_GENERATION.value,
            NodeName.PLAN_EXECUTION.value: NodeName.PLAN_EXECUTION.value,
        },
    )

    return builder.compile(checkpointer=MemorySaver()), table_relation_node, schema_store


def compute_pre_mix_tables(
    tr_node: TableRelationNode,
    schema_store: SchemaStore,
    database_id: str,
    table_docs: list[dict[str, Any]],
    column_docs: list[dict[str, Any]],
    tables_by_id: dict[str, str],
    foreign_keys: list[Any],
) -> tuple[list[str], list[str]]:
    table_scores: dict[str, float] = {}
    for doc in table_docs:
        meta = doc.get("metadata", {})
        tid = meta.get("tableId") or meta.get("table_id")
        score = doc.get("score")
        if tid and score is not None:
            table_scores[str(tid)] = max(table_scores.get(str(tid), 0.0), float(score))

    column_scores: list[ColumnScore] = []
    for doc in column_docs:
        meta = doc.get("metadata", {})
        tid = meta.get("tableId") or meta.get("table_id")
        cid = meta.get("columnId") or meta.get("column_id")
        score = doc.get("score")
        if tid and cid and score is not None:
            column_scores.append(ColumnScore(table_id=str(tid), column_id=str(cid), score=float(score)))

    sel = tr_node.select_threshold(table_scores, column_scores)
    seed_ids = sel.table_ids
    related_fks = [
        fk
        for fk in foreign_keys
        if str(fk.source_column.db_table.id) in seed_ids
        or str(fk.target_column.db_table.id) in seed_ids
    ]
    fk_ids: set[str] = set()
    for fk in related_fks:
        fk_ids.add(str(fk.source_column.db_table.id))
        fk_ids.add(str(fk.target_column.db_table.id))
    final_ids = list(dict.fromkeys([*seed_ids, *fk_ids]))

    seed_names = sorted({tables_by_id[tid] for tid in seed_ids if tid in tables_by_id})
    final_names = sorted({tables_by_id[tid] for tid in final_ids if tid in tables_by_id})
    return seed_names, final_names


def check_evidence_covered(gold_evidence: str, recalled_evidence: str) -> bool:
    ge = (gold_evidence or "").strip()
    if not ge:
        return True
    norm_recalled = re.sub(r"\s+", " ", (recalled_evidence or "").lower())
    clauses = [c.strip() for c in ge.split(";") if c.strip()]
    for clause in clauses:
        norm_c = re.sub(r"\s+", " ", clause.lower())
        if norm_c not in norm_recalled:
            return False
    return True


def evaluate_single_question(
    item: dict[str, Any],
    leave_one_out: bool,
    chat_model: str,
    tables_by_db: dict[str, dict[str, str]],
    fks_by_db: dict[str, list[Any]],
) -> dict[str, Any]:
    t0 = time.time()
    q_id = item["question_id"]
    db_id = item["db_id"]
    q_text = item["question"]
    gold_evidence = item.get("evidence", "") or ""
    gold_sql = item["SQL"]
    difficulty = item.get("difficulty", "unknown")
    gold_tables = sorted(extract_tables_from_sql(gold_sql, dialect="sqlite"))
    gold_table_set = set(gold_tables)

    db_path = locate_sqlite_database(db_id)
    assert db_path is not None, f"SQLite db not found for {db_id}"
    gold_ok, gold_rows, gold_err = execute_sqlite_readonly(db_path, gold_sql)

    graph, tr_node, schema_store = build_eval_graph(chat_model)
    config = {"configurable": {"thread_id": f"eval-{db_id}-{q_id}-loo{int(leave_one_out)}-{int(t0*1000)}"}, "recursion_limit": 30}
    initial_state: dict[str, Any] = {
        "input": q_text,
        "databaseId": db_id,
        StateKey.LEAVE_ONE_OUT.value: leave_one_out,
        StateKey.EXCLUDE_QUESTION_TEXT.value: q_text,
    }

    rewrite_query = ""
    recalled_evidence = ""
    table_docs: list[dict[str, Any]] = []
    column_docs: list[dict[str, Any]] = []
    post_mix_tables: list[str] = []
    feasibility_output = ""
    feasibility_passed = False
    plan_output: Any = None
    sql_attempts: list[str] = []
    sql_exec_errors: list[Optional[str]] = []
    node_error: Optional[str] = None

    try:
        for mode, event in graph.stream(initial_state, config, stream_mode=["updates"]):
            if mode != "updates" or not isinstance(event, dict):
                continue
            for node_name, upd in event.items():
                if not isinstance(upd, dict):
                    continue
                if node_name == NodeName.EVIDENCE_RECALL.value:
                    rewrite_query = upd.get(StateKey.REWRITE_QUERY.value, "") or ""
                    recalled_evidence = upd.get(StateKey.EVIDENCE.value, "") or ""
                elif node_name == NodeName.SCHEMA_RECALL.value:
                    table_docs = upd.get(StateKey.TABLE_SCHEMA.value, []) or []
                    column_docs = upd.get(StateKey.COLUMN_SCHEMA.value, []) or []
                elif node_name == NodeName.TABLE_RELATION.value:
                    rel_out = upd.get(StateKey.TABLE_RELATION.value) or {}
                    db_tables = rel_out.get("dbTables") or rel_out.get("db_tables") or []
                    post_mix_tables = sorted(
                        {str(t.get("name", "")).lower() for t in db_tables if t.get("name")}
                    )
                elif node_name == NodeName.FEASIBILITY_ASSESSMENT.value:
                    feasibility_output = upd.get(StateKey.FEASIBILITY_RESULT.value, "") or ""
                    feasibility_passed = "【需求类型】：《数据分析》" in feasibility_output
                elif node_name == NodeName.PLANNER.value:
                    plan_output = upd.get(StateKey.PLAN.value)
                elif node_name == NodeName.SQL_GENERATION.value:
                    sql_attempts.append(upd.get(StateKey.SQL_GENERATION_RESULT.value, "") or "")
                elif node_name == NodeName.SQL_EXECUTION.value:
                    exec_res = upd.get(StateKey.SQL_EXECUTION_RESULT.value) or {}
                    err_msg = (exec_res.get("resultSet") or {}).get("errorMsg")
                    sql_exec_errors.append(err_msg)
    except Exception as exc:
        err_str = str(exc)
        err_name = type(exc).__name__
        if "insufficient_quota" in err_str or "PermissionDeniedError" in err_name:
            raise
        if "RateLimitError" in err_name or "429" in err_str or "rate_limit" in err_str.lower():
            raise
        node_error = f"{err_name}: {exc}"

    pre_mix_seed_tables, pre_mix_fk_tables = compute_pre_mix_tables(
        tr_node,
        schema_store,
        db_id,
        table_docs,
        column_docs,
        tables_by_db[db_id],
        fks_by_db[db_id],
    )
    pre_mix_all_recalled = gold_table_set.issubset(set(pre_mix_fk_tables))
    post_mix_all_recalled = gold_table_set.issubset(set(post_mix_tables))
    gold_evidence_covered = check_evidence_covered(gold_evidence, recalled_evidence)

    first_attempt_sql = sql_attempts[0] if sql_attempts else None
    final_sql = sql_attempts[-1] if sql_attempts else None
    repair_count = sum(1 for err in sql_exec_errors if err is not None)
    # Note: if final failed too, sql_exec_errors has up to 4 errors (initial + 3 retries),
    # whereas node SQL_REPAIR_COUNT caps at 3 retries.
    repair_triggers = min(repair_count, 3)

    first_exec_ok = False
    first_ex = False
    if first_attempt_sql:
        f_ok, f_rows, _ = execute_sqlite_readonly(db_path, first_attempt_sql)
        first_exec_ok = f_ok
        if f_ok and gold_ok:
            first_ex, _ = compute_ex(f_rows, gold_rows)

    final_exec_ok = False
    final_ex = False
    is_empty_match = False
    pred_rows: Optional[list[tuple[Any, ...]]] = None
    pred_err: Optional[str] = None
    if final_sql:
        p_ok, pred_rows, pred_err = execute_sqlite_readonly(db_path, final_sql)
        final_exec_ok = p_ok
        if p_ok and gold_ok:
            final_ex, is_empty_match = compute_ex(pred_rows, gold_rows)

    # 5-stage error attribution funnel
    if final_ex:
        error_stage = "passed"
    elif not feasibility_passed:
        error_stage = "1_feasibility_rejected"
    elif not post_mix_all_recalled:
        error_stage = "2_table_recall_missed"
    elif not gold_evidence_covered:
        error_stage = "3_evidence_missed"
    elif not final_exec_ok:
        error_stage = "4_sql_exec_error_after_repairs"
    else:
        error_stage = "5_sql_logic_mismatch"

    return {
        "question_id": q_id,
        "db_id": db_id,
        "difficulty": difficulty,
        "question": q_text,
        "gold_evidence": gold_evidence,
        "gold_sql": gold_sql,
        "gold_tables": gold_tables,
        "leave_one_out": leave_one_out,
        "model": chat_model,
        "rewrite_query": rewrite_query,
        "recalled_evidence": recalled_evidence,
        "gold_evidence_covered": gold_evidence_covered,
        "pre_mix_seed_tables": pre_mix_seed_tables,
        "pre_mix_fk_tables": pre_mix_fk_tables,
        "post_mix_tables": post_mix_tables,
        "pre_mix_all_tables_recalled": pre_mix_all_recalled,
        "post_mix_all_tables_recalled": post_mix_all_recalled,
        "feasibility_passed": feasibility_passed,
        "feasibility_output": feasibility_output,
        "plan": plan_output,
        "sql_attempts": sql_attempts,
        "sql_exec_errors": sql_exec_errors,
        "first_attempt_sql": first_attempt_sql,
        "final_sql": final_sql,
        "repair_count": repair_triggers,
        "first_attempt_exec_ok": first_exec_ok,
        "final_exec_ok": final_exec_ok,
        "first_attempt_ex": first_ex,
        "final_ex": final_ex,
        "is_empty_match": is_empty_match,
        "rescued_by_repair": (not first_ex) and final_ex,
        "exec_rescued_by_repair": (not first_exec_ok) and final_exec_ok,
        "error_stage": error_stage,
        "gold_exec_ok": gold_ok,
        "gold_row_count": len(gold_rows) if gold_rows is not None else None,
        "pred_row_count": len(pred_rows) if pred_rows is not None else None,
        "pred_exec_error": pred_err,
        "node_error": node_error,
        "elapsed_s": round(time.time() - t0, 2),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    ex_count = sum(1 for r in records if r["final_ex"])
    empty_match_count = sum(1 for r in records if r["is_empty_match"])
    feas_rejected = sum(1 for r in records if not r["feasibility_passed"])
    pre_mix_recall = sum(1 for r in records if r["pre_mix_all_tables_recalled"])
    post_mix_recall = sum(1 for r in records if r["post_mix_all_tables_recalled"])
    avg_pre_tables = sum(len(r["pre_mix_fk_tables"]) for r in records) / n
    avg_post_tables = sum(len(r["post_mix_tables"]) for r in records) / n

    sql_gen_records = [r for r in records if r["first_attempt_sql"]]
    m = len(sql_gen_records) or 1
    first_exec_ok = sum(1 for r in sql_gen_records if r["first_attempt_exec_ok"])
    final_exec_ok = sum(1 for r in sql_gen_records if r["final_exec_ok"])
    first_ex = sum(1 for r in sql_gen_records if r["first_attempt_ex"])
    final_ex_sql = sum(1 for r in sql_gen_records if r["final_ex"])
    rescued_ex = sum(1 for r in records if r["rescued_by_repair"])
    rescued_exec = sum(1 for r in records if r["exec_rescued_by_repair"])
    repaired_any = sum(1 for r in records if r["repair_count"] > 0)

    by_diff: dict[str, Any] = {}
    for diff in ("simple", "moderate", "challenging"):
        sub = [r for r in records if r["difficulty"] == diff]
        if sub:
            sc = sum(1 for r in sub if r["final_ex"])
            by_diff[diff] = {"count": len(sub), "ex_count": sc, "ex_rate": round(sc / len(sub), 4)}

    funnel_keys = (
        "passed",
        "1_feasibility_rejected",
        "2_table_recall_missed",
        "3_evidence_missed",
        "4_sql_exec_error_after_repairs",
        "5_sql_logic_mismatch",
    )
    funnel = {k: sum(1 for r in records if r["error_stage"] == k) for k in funnel_keys}

    return {
        "total": n,
        "ex_count": ex_count,
        "ex_rate": round(ex_count / n, 4),
        "empty_match_count": empty_match_count,
        "by_difficulty": by_diff,
        "feasibility_false_reject_count": feas_rejected,
        "feasibility_false_reject_rate": round(feas_rejected / n, 4),
        "mix_selector_ablation": {
            "pre_mix_all_tables_recalled_count": pre_mix_recall,
            "pre_mix_all_tables_recalled_rate": round(pre_mix_recall / n, 4),
            "post_mix_all_tables_recalled_count": post_mix_recall,
            "post_mix_all_tables_recalled_rate": round(post_mix_recall / n, 4),
            "avg_pre_mix_tables": round(avg_pre_tables, 4),
            "avg_post_mix_tables": round(avg_post_tables, 4),
        },
        "sql_self_repair_ablation": {
            "sql_generated_count": len(sql_gen_records),
            "first_attempt_exec_ok_count": first_exec_ok,
            "first_attempt_exec_ok_rate": round(first_exec_ok / m, 4),
            "final_exec_ok_count": final_exec_ok,
            "final_exec_ok_rate": round(final_exec_ok / m, 4),
            "first_attempt_ex_count": first_ex,
            "first_attempt_ex_rate": round(first_ex / n, 4),
            "final_ex_count": final_ex_sql,
            "final_ex_rate": round(final_ex_sql / n, 4),
            "repair_triggered_count": repaired_any,
            "exec_rescued_by_repair_count": rescued_exec,
            "ex_rescued_by_repair_count": rescued_ex,
        },
        "error_funnel": funnel,
        "node_errors": sum(1 for r in records if r["node_error"]),
    }


def load_full_cs_questions() -> list[dict[str, Any]]:
    with DEV_JSON_PATH.open("r", encoding="utf-8") as f:
        all_dev = json.load(f)
    cs_all = [q for q in all_dev if q.get("db_id") == "california_schools"]
    cs_all.sort(key=lambda x: x["question_id"])
    return cs_all


def run_group(
    group_name: str,
    questions: list[dict[str, Any]],
    leave_one_out: bool,
    chat_model: str,
    workers: int,
    tables_by_db: dict[str, dict[str, str]],
    fks_by_db: dict[str, list[Any]],
    out_jsonl: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t0 = time.time()
    print(f"\n>>> Starting {group_name} ({len(questions)} Qs, leave_one_out={leave_one_out}, model={chat_model}, workers={workers})")
    records: list[dict[str, Any]] = []
    retry_items: list[dict[str, Any]] = []
    quota_exhausted = False

    def _run_batch(batch: list[dict[str, Any]], max_workers: int, allow_rate_retry: bool) -> list[dict[str, Any]]:
        nonlocal quota_exhausted
        failed_rate: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    evaluate_single_question,
                    item,
                    leave_one_out,
                    chat_model,
                    tables_by_db,
                    fks_by_db,
                ): item
                for item in batch
            }
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    rec = fut.result()
                    records.append(rec)
                    done_cnt = len(records)
                    if done_cnt % 3 == 0 or done_cnt == len(questions):
                        cur_ex = sum(1 for r in records if r["final_ex"])
                        print(
                            f"  [{group_name}] {done_cnt}/{len(questions)} done | "
                            f"EX={cur_ex}/{done_cnt} ({cur_ex/done_cnt:.1%}) | {time.time()-t0:.1f}s"
                        )
                except Exception as exc:
                    err_str = str(exc)
                    err_name = type(exc).__name__
                    if "insufficient_quota" in err_str or "PermissionDeniedError" in err_name:
                        print(f"  [{group_name}] Quota exhausted on Q{item['question_id']}: {err_str[:160]}")
                        quota_exhausted = True
                    elif allow_rate_retry and ("RateLimitError" in err_name or "429" in err_str or "rate_limit" in err_str.lower()):
                        print(f"  [{group_name}] Rate limit on Q{item['question_id']}, queuing retry at lower concurrency")
                        failed_rate.append(item)
                    else:
                        print(f"  [{group_name}] Error on Q{item['question_id']}: {err_name}: {err_str[:160]}")
        return failed_rate

    retry_items = _run_batch(questions, workers, allow_rate_retry=True)
    if retry_items and not quota_exhausted:
        retry_workers = min(3, workers)
        print(f"  [{group_name}] Retrying {len(retry_items)} rate-limited questions with workers={retry_workers}...")
        time.sleep(3.0)
        still_failed = _run_batch(retry_items, retry_workers, allow_rate_retry=False)
        if still_failed:
            print(f"  [{group_name}] {len(still_failed)} questions still failed after retry.")

    records.sort(key=lambda r: r["question_id"])
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = summarize_records(records)
    summary["group_name"] = group_name
    summary["quota_exhausted"] = quota_exhausted
    summary["elapsed_seconds"] = round(time.time() - t0, 2)
    if records:
        print(f"<<< Finished {group_name}: EX={summary['ex_count']}/{summary['total']} ({summary['ex_rate']:.2%}) in {summary['elapsed_seconds']}s")
    return records, summary


def preload_schema_metadata() -> tuple[dict[str, dict[str, str]], dict[str, list[Any]]]:
    store = SchemaStore()
    tables_by_db: dict[str, dict[str, str]] = {}
    fks_by_db: dict[str, list[Any]] = {}
    for db_id in ("california_schools", "debit_card_specializing"):
        sql = """
            SELECT
                t.id as table_id, t.name as table_name, t.description as table_desc,
                c.id as col_id, c.name as col_name, c.type as col_type,
                c.description as col_desc, c.is_primary_key as col_pk
            FROM db_table t
            LEFT JOIN db_column c ON t.id = c.table_id
            WHERE t.database_id = %(database_id)s
            ORDER BY t.name, c.name
        """
        tables = store._fetch_tables_with_columns(sql, {"database_id": db_id})
        tables_by_db[db_id] = {str(t.id): t.name.lower() for t in tables}
        fks_by_db[db_id] = store.find_foreign_keys_by_database_id(db_id)
    return tables_by_db, fks_by_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2E Text2SQL evaluation suite.")
    parser.add_argument("--phase", choices=["baseline", "postfix", "all", "holdout_prefix", "full_glm5", "holdout_glm51"], default="baseline")
    parser.add_argument("--model", default=settings.CHAT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-group-a", action="store_true")
    parser.add_argument("--resume-cs", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cs_sample, dc_holdout = load_eval_splits()
    tables_by_db, fks_by_db = preload_schema_metadata()

    if args.phase == "full_glm5":
        cs_full = load_full_cs_questions()
        cs_sample_ids = {q["question_id"] for q in cs_sample}
        glm5_summary: dict[str, Any] = {
            "config": {
                "chat_model": args.model,
                "california_schools_full_size": len(cs_full),
                "california_schools_slice_size": len(cs_sample),
                "california_schools_slice_ids": sorted(cs_sample_ids),
                "debit_card_holdout_size": len(dc_holdout),
            }
        }
        cs_jsonl = RESULTS_DIR / "e2e_glm5_full_cs_89.jsonl"
        if args.resume_cs and cs_jsonl.exists():
            cs_records = [json.loads(line) for line in cs_jsonl.open("r", encoding="utf-8")]
            cs_full_sum = summarize_records(cs_records)
            cs_full_sum["group_name"] = "glm5_full_california_schools_89"
            cs_full_sum["quota_exhausted"] = False
            print(f"Resumed {len(cs_records)} records from {cs_jsonl}")
        else:
            cs_records, cs_full_sum = run_group(
                "glm5_full_california_schools_89",
                cs_full,
                leave_one_out=True,
                chat_model=args.model,
                workers=args.workers,
                tables_by_db=tables_by_db,
                fks_by_db=fks_by_db,
                out_jsonl=cs_jsonl,
            )
        glm5_summary["full_california_schools_89"] = cs_full_sum

        slice_records = [r for r in cs_records if r["question_id"] in cs_sample_ids]
        slice_sum = summarize_records(slice_records)
        slice_sum["group_name"] = "glm5_slice_california_schools_30"
        glm5_summary["slice_california_schools_30"] = slice_sum

        if not cs_full_sum.get("quota_exhausted"):
            _, holdout_sum = run_group(
                "glm5_holdout_debit_card_15",
                dc_holdout,
                leave_one_out=True,
                chat_model=args.model,
                workers=args.workers,
                tables_by_db=tables_by_db,
                fks_by_db=fks_by_db,
                out_jsonl=RESULTS_DIR / "e2e_glm5_holdout_15.jsonl",
            )
            glm5_summary["holdout_debit_card_15"] = holdout_sum

        glm5_summary_path = RESULTS_DIR / "e2e_eval_glm5_summary.json"
        glm5_summary_path.write_text(json.dumps(glm5_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n=== GLM-5 Full Evaluation Summary Saved ===")
        print(json.dumps(glm5_summary, ensure_ascii=False, indent=2))
        return

    if args.phase == "holdout_glm51":
        _, holdout_sum = run_group(
            "glm51_holdout_debit_card_15",
            dc_holdout,
            leave_one_out=True,
            chat_model=args.model,
            workers=args.workers,
            tables_by_db=tables_by_db,
            fks_by_db=fks_by_db,
            out_jsonl=RESULTS_DIR / "e2e_glm51_holdout_15.jsonl",
        )
        glm5_summary_path = RESULTS_DIR / "e2e_eval_glm5_summary.json"
        glm5_summary = json.loads(glm5_summary_path.read_text(encoding="utf-8")) if glm5_summary_path.exists() else {}
        glm5_summary["holdout_debit_card_15_glm51"] = holdout_sum
        glm5_summary_path.write_text(json.dumps(glm5_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n=== GLM-5.1 Holdout Summary Merged ===")
        print(json.dumps(holdout_sum, ensure_ascii=False, indent=2))
        return

    summary_path = RESULTS_DIR / "e2e_eval_summary.json"
    all_summaries: dict[str, Any] = {}
    if summary_path.exists():
        try:
            all_summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            all_summaries = {}

    all_summaries["config"] = {
        "chat_model": args.model,
        "california_schools_sample_size": len(cs_sample),
        "california_schools_difficulty": dict(Counter(q["difficulty"] for q in cs_sample)),
        "debit_card_holdout_size": len(dc_holdout),
        "debit_card_difficulty": dict(Counter(q["difficulty"] for q in dc_holdout)),
    }

    if args.phase == "holdout_prefix":
        _, sum_h_pre = run_group(
            "holdout_pre_fix",
            dc_holdout,
            leave_one_out=True,
            chat_model=args.model,
            workers=args.workers,
            tables_by_db=tables_by_db,
            fks_by_db=fks_by_db,
            out_jsonl=RESULTS_DIR / "e2e_holdout_prefix_15.jsonl",
        )
        all_summaries["holdout_pre_fix"] = sum_h_pre

    if args.phase in ("baseline", "all"):
        if not args.skip_group_a:
            _, sum_a = run_group(
                "group_a_with_leakage",
                cs_sample,
                leave_one_out=False,
                chat_model=args.model,
                workers=args.workers,
                tables_by_db=tables_by_db,
                fks_by_db=fks_by_db,
                out_jsonl=RESULTS_DIR / "e2e_group_a_leakage_30.jsonl",
            )
            all_summaries["group_a_with_leakage"] = sum_a

        _, sum_b = run_group(
            "group_b_leave_one_out_baseline",
            cs_sample,
            leave_one_out=True,
            chat_model=args.model,
            workers=args.workers,
            tables_by_db=tables_by_db,
            fks_by_db=fks_by_db,
            out_jsonl=RESULTS_DIR / "e2e_group_b_baseline_30.jsonl",
        )
        all_summaries["group_b_baseline"] = sum_b

        _, sum_h_pre = run_group(
            "holdout_pre_fix",
            dc_holdout,
            leave_one_out=True,
            chat_model=args.model,
            workers=args.workers,
            tables_by_db=tables_by_db,
            fks_by_db=fks_by_db,
            out_jsonl=RESULTS_DIR / "e2e_holdout_prefix_15.jsonl",
        )
        all_summaries["holdout_pre_fix"] = sum_h_pre

    if args.phase in ("postfix", "all"):
        _, sum_b_post = run_group(
            "group_b_leave_one_out_postfix",
            cs_sample,
            leave_one_out=True,
            chat_model=args.model,
            workers=args.workers,
            tables_by_db=tables_by_db,
            fks_by_db=fks_by_db,
            out_jsonl=RESULTS_DIR / "e2e_group_b_postfix_30.jsonl",
        )
        all_summaries["group_b_postfix"] = sum_b_post

        _, sum_h_post = run_group(
            "holdout_post_fix",
            dc_holdout,
            leave_one_out=True,
            chat_model=args.model,
            workers=args.workers,
            tables_by_db=tables_by_db,
            fks_by_db=fks_by_db,
            out_jsonl=RESULTS_DIR / "e2e_holdout_postfix_15.jsonl",
        )
        all_summaries["holdout_post_fix"] = sum_h_post

    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== E2E Evaluation Summary Saved ===")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
