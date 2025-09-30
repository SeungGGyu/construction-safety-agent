from core.agentstate import AgentState

def websearch(state: AgentState):
    # 여기서는 placeholder: 기존 retriever로 다른 인덱스 or 확장 쿼리 재검색
    q = state.get("query") or state["messages"][-1].content
    docs_web = retriever_instance.retrieve(q + " 안전기준 법규 조항 체크리스트")  # 예시: 확장 키워드
    # 웹 보강 이후에는 더 이상 웹으로 빠지지 않게 플래그 OFF
    new_docs = (state.get("retrieved") or []) + docs_web
    return {
        "retrieved": new_docs,
        "selected": new_docs,  # 간단히 동일 적용(원하면 grade_documents로 다시 필터)
        "web_fallback": False  # 다음 grade에서 finalize로 수렴하게끔
    }
