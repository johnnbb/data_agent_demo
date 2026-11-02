import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional
from openai import OpenAI

from app.config import settings
from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.prompts.manager import prompt_manager
from app.retrieval.schema_models import (
    DbForeignKeySchemaView,
    DbTableSchemaView,
    Schema,
)
from app.retrieval.schema_store import SchemaStore

logger = logging.getLogger(__name__)


@dataclass
class ColumnScore:
    table_id: str
    column_id: str
    score: float


@dataclass
class ThresholdSelection:
    threshold: float
    table_ids: list[str]
    high_similarity_column_ids_by_table: dict[str, set[str]]
    high_similarity_table_count: int
    high_similarity_column_table_count: int
    column_table_score_keys: int


class TableRelationNode:
    """
    表关联拓扑补齐与精筛节点（对应 Kotlin 的 TableRelationNode.kt）。
    通过动态阈值过滤、外键拓扑扩展、模式构建与大模型二次精筛 (Mix Selector) 消除外键断裂。
    """

    def __init__(
        self,
        schema_store: Optional[SchemaStore] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        chat_model: Optional[str] = None,
    ):
        self.schema_store = schema_store or SchemaStore()
        self.chat_model = chat_model or settings.CHAT_MODEL
        self.client = OpenAI(
            api_key=openai_api_key or settings.OPENAI_API_KEY,
            base_url=openai_base_url or settings.OPENAI_BASE_URL,
        )
        self.base_high_similarity_threshold = 0.4
        self.target_table_count = 4

    def select_threshold(
        self,
        table_scores: dict[str, float],
        column_scores: list[ColumnScore],
    ) -> ThresholdSelection:
        """
        基于得分的动态阈值算法 (Dynamic Thresholding)。
        在相关性（0.6~0.99）和上下文长度（目标4张表）之间取得平衡。
        """
        def build_selection(threshold: float) -> ThresholdSelection:
            table_score_from_column_recall: dict[str, float] = {}
            for cs in column_scores:
                table_score_from_column_recall[cs.table_id] = max(
                    table_score_from_column_recall.get(cs.table_id, 0.0), cs.score
                )

            high_similarity_table_ids = [
                tid for tid, score in table_scores.items() if score >= threshold
            ]
            high_similarity_column_scores = [cs for cs in column_scores if cs.score >= threshold]
            high_similarity_column_table_ids = {cs.table_id for cs in high_similarity_column_scores}
            high_similarity_column_ids_by_table: dict[str, set[str]] = {}
            for cs in high_similarity_column_scores:
                high_similarity_column_ids_by_table.setdefault(cs.table_id, set()).add(cs.column_id)

            # 保持插入顺序的集合
            selected_table_ids = dict.fromkeys(high_similarity_table_ids)
            for tid in high_similarity_column_table_ids:
                selected_table_ids[tid] = True

            # 兜底：如果基准阈值下无任何表入选，取表相似度 Top 1 和字段相似度所属表的 Top 1
            if not selected_table_ids:
                if table_scores:
                    top_table = max(table_scores.items(), key=lambda x: x[1])[0]
                    selected_table_ids[top_table] = True
                if table_score_from_column_recall:
                    top_col_table = max(table_score_from_column_recall.items(), key=lambda x: x[1])[0]
                    selected_table_ids[top_col_table] = True

            return ThresholdSelection(
                threshold=threshold,
                table_ids=list(selected_table_ids.keys()),
                high_similarity_column_ids_by_table=high_similarity_column_ids_by_table,
                high_similarity_table_count=len(high_similarity_table_ids),
                high_similarity_column_table_count=len(high_similarity_column_table_ids),
                column_table_score_keys=len(table_score_from_column_recall),
            )

        best = build_selection(self.base_high_similarity_threshold)
        if len(best.table_ids) <= self.target_table_count:
            return best

        for i in range(1, 30):
            threshold = min(self.base_high_similarity_threshold + i * 0.01, 0.99)
            candidate = build_selection(threshold)
            candidate_diff = abs(self.target_table_count - len(candidate.table_ids))
            best_diff = abs(self.target_table_count - len(best.table_ids))
            if candidate_diff < best_diff or (
                candidate_diff == best_diff and candidate.threshold > best.threshold
            ):
                best = candidate

        return best

    def __call__(self, state: AgentState) -> dict[str, Any]:
        rewrite_query = state.get(StateKey.REWRITE_QUERY.value) or state.get("input", "")
        evidence = state.get(StateKey.EVIDENCE.value, "")
        table_documents = state.get(StateKey.TABLE_SCHEMA.value, [])
        column_documents = state.get(StateKey.COLUMN_SCHEMA.value, [])
        database_id = state.get("databaseId", "")

        # 1. 抽取表和字段的向量检索得分
        table_scores: dict[str, float] = {}
        for doc in table_documents:
            metadata = doc.get("metadata", {})
            table_id = metadata.get("tableId") or metadata.get("table_id")
            score = doc.get("score")
            if table_id and score is not None:
                tid_str = str(table_id)
                table_scores[tid_str] = max(table_scores.get(tid_str, 0.0), float(score))

        column_scores: list[ColumnScore] = []
        for doc in column_documents:
            metadata = doc.get("metadata", {})
            table_id = metadata.get("tableId") or metadata.get("table_id")
            column_id = metadata.get("columnId") or metadata.get("column_id")
            score = doc.get("score")
            if table_id and column_id and score is not None:
                column_scores.append(
                    ColumnScore(
                        table_id=str(table_id),
                        column_id=str(column_id),
                        score=float(score),
                    )
                )

        # 2. 动态阈值筛选种子表
        threshold_selection = self.select_threshold(table_scores, column_scores)
        seed_table_ids = threshold_selection.table_ids
        high_sim_col_ids_by_table = threshold_selection.high_similarity_column_ids_by_table

        # 3. 外键拓扑延伸 (Topology Expansion)
        foreign_keys = self.schema_store.find_foreign_keys_by_database_id(database_id)
        related_foreign_keys = [
            fk
            for fk in foreign_keys
            if str(fk.source_column.db_table.id) in seed_table_ids
            or str(fk.target_column.db_table.id) in seed_table_ids
        ]

        foreign_key_related_table_ids: set[str] = set()
        for fk in related_foreign_keys:
            foreign_key_related_table_ids.add(str(fk.source_column.db_table.id))
            foreign_key_related_table_ids.add(str(fk.target_column.db_table.id))

        final_table_ids_dict = dict.fromkeys(seed_table_ids)
        for tid in foreign_key_related_table_ids:
            final_table_ids_dict[tid] = True
        final_table_ids = list(final_table_ids_dict.keys())

        foreign_key_column_ids_by_table: dict[str, set[str]] = {}
        for fk in related_foreign_keys:
            s_tid = str(fk.source_column.db_table.id)
            s_cid = str(fk.source_column.id)
            foreign_key_column_ids_by_table.setdefault(s_tid, set()).add(s_cid)

            t_tid = str(fk.target_column.db_table.id)
            t_cid = str(fk.target_column.id)
            foreign_key_column_ids_by_table.setdefault(t_tid, set()).add(t_cid)

        # 4. 查询展开后的完整表和列元数据，并根据高分列+外键列进行列裁剪
        tables = self.schema_store.find_tables_by_ids(database_id, final_table_ids)
        candidate_tables: list[DbTableSchemaView] = []
        for table in tables:
            tid = str(table.id)
            high_cols = high_sim_col_ids_by_table.get(tid, set())
            fk_cols = foreign_key_column_ids_by_table.get(tid, set())
            selected_cols = high_cols.union(fk_cols)
            if selected_cols:
                filtered_columns = [c for c in table.columns if str(c.id) in selected_cols]
            else:
                filtered_columns = table.columns
            candidate_tables.append(table.model_copy(update={"columns": filtered_columns}))

        logger.info(
            f"TableRelationNode merged tables: direct={len(table_scores)}, "
            f"highTable={threshold_selection.high_similarity_table_count}, "
            f"highColumn={threshold_selection.high_similarity_column_table_count}, "
            f"threshold={threshold_selection.threshold:.2f}, "
            f"fkExpanded={len(foreign_key_related_table_ids)}, merged={len(final_table_ids)}"
        )

        # 5. 组装 Mix Selector Prompt 供大模型二次精筛
        candidate_schema = Schema(
            database_id=database_id,
            db_tables=candidate_tables,
            db_foreign_keys=related_foreign_keys,
        )
        prompt = prompt_manager.render(
            "mix-selector.st",
            schema_info=candidate_schema.build_scheme_prompt(),
            question=rewrite_query,
            evidence=evidence,
        )

        logger.info(f"Mix selector prompt:\n{prompt}")

        # 6. 调用大模型进行表筛选
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )
        content = resp.choices[0].message.content or "[]"
        clean_content = content.strip()
        if clean_content.startswith("```"):
            clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content)
            clean_content = re.sub(r"\s*```$", "", clean_content)

        try:
            filter_table_names: list[str] = json.loads(clean_content)
        except Exception as e:
            logger.warning(f"Failed to parse mix selector output as JSON: {clean_content}, error: {e}")
            # 降级：如果大模型输出异常，保留候选集中的所有表名
            filter_table_names = [t.name for t in candidate_tables]

        # 7. 根据精筛结果查询最终表（恢复完整列）与过滤外键
        filter_tables = self.schema_store.find_tables_by_database_id_and_names(
            database_id, filter_table_names
        )
        filter_foreign_keys = [
            fk
            for fk in foreign_keys
            if (fk.source_column.db_table and fk.source_column.db_table.name in filter_table_names)
            or (fk.target_column.db_table and fk.target_column.db_table.name in filter_table_names)
        ]

        logger.info(f"Mix selector filter tables: {filter_table_names}")

        final_schema = Schema(
            database_id=database_id,
            db_tables=filter_tables,
            db_foreign_keys=filter_foreign_keys,
        )

        return {
            StateKey.TABLE_RELATION.value: final_schema.model_dump(by_alias=True)
        }


table_relation_node = TableRelationNode()

