import json
import logging
from typing import Any, Optional
import psycopg
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.prompts.manager import prompt_manager
from app.retrieval.retriever import VectorRetriever

logger = logging.getLogger(__name__)


class EvidenceQueryRewrite(BaseModel):
    standalone_query: str = Field(description="重写后的完整独立句子，消除代词与噪音")


class EvidenceRecallNode:
    """
    业务领域知识与历史问答双路召回节点。
    """

    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
        db_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        chat_model: Optional[str] = None,
    ):
        self.retriever = retriever or VectorRetriever()
        self.db_url = db_url or settings.DATABASE_URL
        self.chat_model = chat_model or settings.CHAT_MODEL
        self.client = OpenAI(
            api_key=openai_api_key or settings.OPENAI_API_KEY,
            base_url=openai_base_url or settings.OPENAI_BASE_URL,
        )

    def rewrite_query(self, user_input: str, multi_turn: str = "(无)") -> str:
        format_instruction = json.dumps({"standalone_query": "重写后的句子"}, ensure_ascii=False)
        prompt = prompt_manager.render_evidence_query_rewrite(
            latest_query=user_input,
            format_str=format_instruction,
            multi_turn=multi_turn,
        )

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
            parsed = EvidenceQueryRewrite(**data)
            return parsed.standalone_query
        except Exception as e:
            logger.warning(f"Failed to parse query rewrite response: {content}, error: {e}")
            return user_input

    def fetch_question_knowledge_answers(self, knowledge_ids: list[str]) -> list[dict[str, Any]]:
        if not knowledge_ids:
            return []

        valid_ids = list({str(kid) for kid in knowledge_ids if kid})
        if not valid_ids:
            return []

        sql = """
            SELECT id, question, answer, database_id
            FROM question_knowledge
            WHERE id = ANY(%s::uuid[]);
        """
        records = []
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (valid_ids,))
                for row in cur.fetchall():
                    records.append({
                        "id": str(row[0]),
                        "question": row[1],
                        "answer": row[2],
                        "database_id": row[3],
                    })
        return records

    def __call__(self, state: AgentState) -> dict[str, Any]:
        user_input = state.get("input", "")
        database_id = state.get("databaseId", "")
        multi_turn = state.get(StateKey.MULTI_TURN_CONTEXT) or "(无)"

        # 1. 独立问题重写
        rewrite_query = self.rewrite_query(user_input, multi_turn=multi_turn)
        logger.info(f"Query rewrite: '{user_input}' -> '{rewrite_query}'")

        # 2. 检索业务术语
        glossary_results = self.retriever.search_glossary(rewrite_query, database_id, top_k=4)
        glossary_texts = [r.content for r in glossary_results if r.content]
        glossary_knowledge_text = "\n".join(glossary_texts)

        # 3. 检索历史问答
        question_results = self.retriever.search_questions(rewrite_query, database_id, top_k=4)
        knowledge_ids = [
            r.metadata.get("knowledgeId")
            for r in question_results
            if r.metadata.get("knowledgeId")
        ]

        # 4. 反查实体表
        full_questions = self.fetch_question_knowledge_answers(knowledge_ids)
        question_knowledge_text = "\n".join(
            f"来源：{q['database_id']} Q: {q['question']} A: {q['answer']}"
            for q in full_questions
        )

        # 5. 组装 Evidence
        glossary_prompt = prompt_manager.render_business_knowledge(glossary_knowledge_text)
        knowledge_prompt = prompt_manager.render_agent_knowledge(question_knowledge_text)

        if not glossary_knowledge_text and not question_knowledge_text:
            evidence = "无"
        else:
            evidence = f"{glossary_prompt}\n{knowledge_prompt}"

        # 保持与前端 REWRITE_QUERY 和 EVIDENCE 对齐
        return {
            StateKey.REWRITE_QUERY.value: rewrite_query,
            StateKey.EVIDENCE.value: evidence,
        }


evidence_recall_node = EvidenceRecallNode()
