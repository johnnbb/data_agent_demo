from app.graph.state import AgentState

def route_node(state: AgentState) -> dict:
    user_input = state.get("input", "")
    travel_keywords = ["旅游", "旅行", "出游", "攻略", "景点", "酒店", "美食", "周末去哪"]
    is_travel = any(k in user_input for k in travel_keywords)
    scene = "travel" if is_travel else "study"
    scene_label = "旅行攻略" if is_travel else "学习计划"
    return {
        "scene": scene,
        "sceneLabel": scene_label,
    }
