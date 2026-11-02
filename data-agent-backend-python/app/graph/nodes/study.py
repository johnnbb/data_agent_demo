from langchain_core.messages import SystemMessage, HumanMessage
from app.graph.nodes.travel import get_llm
from app.graph.state import AgentState

async def study_plan_node(state: AgentState) -> dict:
    llm = get_llm()
    user_input = state.get("input", "")
    messages = [
        SystemMessage(content="你是一个适合教程演示的学习教练。输出要口语化、结构清晰，分成目标拆解、今日安排、避坑提醒三部分。"),
        HumanMessage(content=f"请根据这段需求给我一个简短学习计划：{user_input}")
    ]
    response = await llm.ainvoke(messages)
    return {"draft": str(response.content)}
