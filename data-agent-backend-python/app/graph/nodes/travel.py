from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import settings
from app.graph.state import AgentState

def get_llm():
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        streaming=True,
        temperature=0.7,
        extra_body={"enable_thinking": False},
    )

async def travel_plan_node(state: AgentState) -> dict:
    llm = get_llm()
    user_input = state.get("input", "")
    messages = [
        SystemMessage(content="你是一个适合教程演示的旅行助手。输出要口语化、结构清晰，分成目的地亮点、半日安排、注意事项三部分。"),
        HumanMessage(content=f"请根据这段需求给我一个简短旅行攻略：{user_input}")
    ]
    response = await llm.ainvoke(messages)
    return {"draft": str(response.content)}
