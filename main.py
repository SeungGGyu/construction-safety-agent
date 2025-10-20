# main.py
import argparse
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# === Core Imports ===
from core.agentstate import AgentState
from core.retriever import retrieve_node
from core.confirm_retrieval import confirm_retrieval
from core.generate import generate
from core.rewrite import rewrite
from core.websearch import websearch
from core.finalize_response import finalize_response
from core.generation_grader import grade_generation
from core.query import query
from core.final_report import generate_accident_report


# === LangGraph 구성 ===
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("confirm_retrieval", confirm_retrieval)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite)
graph.add_node("websearch", websearch)
graph.add_node("finalize_response", finalize_response)

# --- 흐름 구성 ---
graph.set_entry_point("retrieve")

# 검색 후 사용자 확인 단계로 이동
graph.add_edge("retrieve", "confirm_retrieval")

# 사용자 판단(yes/no)에 따라 다음 단계 결정
graph.add_conditional_edges(
    "confirm_retrieval",
    lambda s: s.get("route", "generate"),
    {
        "generate": "generate",
        "rewrite": "rewrite"
    },
)

# 이후 standard RAG 루프
graph.add_edge("rewrite", "retrieve")
graph.add_edge("websearch", "generate")
graph.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "generate": "generate",              # 환각 → 재생성
        "rewrite": "rewrite",                # 무관 → 리라이트
        "websearch": "websearch",            # 반복 초과 → 웹보강
        "finalize_response": "finalize_response",  # 적합 → 종료
    },
)
graph.add_edge("finalize_response", END)

# === 그래프 컴파일 ===
app = graph.compile()

# === 초기 입력 ===
init_question = query[6]  # 원하는 질의 인덱스 선택
init_state: AgentState = {
    "messages": [HumanMessage(content=init_question)],
    "query": init_question,
    "retries": 0,
    "web_fallback": True,
}

# === 그래프 실행 ===
final_state = app.invoke(init_state)

rag_output = final_state["messages"][-1].content
print("\n=== 최종 응답 ===\n")
print(rag_output)

# === 보고서 생성 ===
report = generate_accident_report(rag_output)
print("\n===== 건설 사고 재발 방지 대책 보고서 초안 =====\n")
print(report)
