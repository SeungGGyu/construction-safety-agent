from core.agentstate import AgentState

def finalize_response(state: AgentState):
    # 마지막 AIMessage가 최종 응답
    return {"messages": state["messages"]}