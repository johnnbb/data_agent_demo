import logging
from typing import Optional
import psycopg
from openai import OpenAI

from app.config import settings
from app.retrieval.models import VectorType, RetrievalResult

logger = logging.getLogger(__name__)

class VectorRetriever:
    """
    轻量高性能 PostgreSQL + pgvector 检索器。
    基于原生 SQL 执行带 metadata 过滤的余弦相似度检索。
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.db_url = db_url or settings.DATABASE_URL
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.client = OpenAI(
            api_key=openai_api_key or settings.OPENAI_API_KEY,
            base_url=openai_base_url or settings.OPENAI_BASE_URL,
        )

    def embed_query(self, query: str) -> list[float]:
        """调用 Embedding API 获取输入文本的 1024 维高维向量"""
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        return resp.data[0].embedding

    def search(
        self,
        query: str,
        database_id: str,
        vector_type: VectorType | str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        核心检索方法：带 metadata 过滤的余弦相似度检索
        """
        # 1. 计算 Query 向量
        q_vec = self.embed_query(query)
        vec_str = "[" + ",".join(str(x) for x in q_vec) + "]"
        type_str = vector_type.value if isinstance(vector_type, VectorType) else str(vector_type)

        # 2. 直连 PostgreSQL 执行 HNSW 索引余弦距离搜索
        sql = """
            SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM vector_store
            WHERE metadata->>'vectorType' = %s
              AND metadata->>'databaseId' = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        results: list[RetrievalResult] = []
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_str, type_str, database_id, vec_str, top_k))
                rows = cur.fetchall()
                for row in rows:
                    results.append(
                        RetrievalResult(
                            content=row[0],
                            metadata=row[1] if isinstance(row[1], dict) else {},
                            score=float(row[2]),
                        )
                    )

        # 按相似度降序排列
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def search_tables(self, query: str, database_id: str, top_k: int = 3) -> list[RetrievalResult]:
        """检索最相关的表结构描述"""
        return self.search(query, database_id, VectorType.TABLE, top_k=top_k)

    def search_columns(self, query: str, database_id: str, top_k: int = 10) -> list[RetrievalResult]:
        """检索最相关的列字段描述"""
        return self.search(query, database_id, VectorType.COLUMN, top_k=top_k)

    def search_glossary(self, query: str, database_id: str, top_k: int = 3) -> list[RetrievalResult]:
        """检索相关的业务口径与术语 (Evidence)"""
        return self.search(query, database_id, VectorType.GLOSSARY_KNOWLEDGE, top_k=top_k)

    def search_questions(self, query: str, database_id: str, top_k: int = 3) -> list[RetrievalResult]:
        """检索历史上最相似的问答范例 (Few-Shot 示例)"""
        return self.search(query, database_id, VectorType.QUESTION_KNOWLEDGE, top_k=top_k)
