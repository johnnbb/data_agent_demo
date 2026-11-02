import logging
from typing import Any, Optional

from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.retrieval.retriever import VectorRetriever

logger = logging.getLogger(__name__)


class SchemeRecallNode:
    """
    数据库元数据结构召回节点（对应 Kotlin 的 SchemeReCallNode.kt）。
    """

    def __init__(self, retriever: Optional[VectorRetriever] = None):
        self.retriever = retriever or VectorRetriever()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        # 优先读取已重写的查询，若无则使用原始输入
        rewrite_query = state.get(StateKey.REWRITE_QUERY) or state.get("input", "")
        database_id = state.get("databaseId", "")

        logger.info(f"SchemeRecallNode searching schema for query: '{rewrite_query}', db: '{database_id}'")

        # 1. 召回相关的表结构 (Top 10)
        table_results = self.retriever.search_tables(
            query=rewrite_query,
            database_id=database_id,
            top_k=10,
        )

        # 2. 召回相关的字段描述 (Top 30)
        column_results = self.retriever.search_columns(
            query=rewrite_query,
            database_id=database_id,
            top_k=30,
        )

        # 构造对齐前端 Document 结构（前端 card 期望字段为 doc.text 或 doc.content，以及 doc.score）
        table_schema = [
            {
                "text": r.content,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in table_results
        ]
        column_schema = [
            {
                "text": r.content,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in column_results
        ]

        logger.info(f"SchemeRecallNode recalled {len(table_schema)} tables and {len(column_schema)} columns.")

        # 与前端 DATA_AGENT_STATE_KEY_RECALL.TABLE_SCHEMA / COLUMN_SCHEMA 对齐 (TABLE_SCHEME / COLUMN_SCHEME)
        return {
            StateKey.TABLE_SCHEMA.value: table_schema,
            StateKey.COLUMN_SCHEMA.value: column_schema,
        }


scheme_recall_node = SchemeRecallNode()
