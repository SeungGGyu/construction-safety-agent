# main.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from core.agentstate import AgentState
from core.retriever import retrieve_node
from core.generate import generate
from core.rewrite import rewrite
from core.websearch import websearch
from core.finalize_response import finalize_response
from core.generation_grader import grade_generation
from core.final_report import generate_accident_report_node
from core.report_grader import grade_report_quality
from core.confirm_retrieval import confirm_retrieval
from typing import Dict, Any, List
from langchain_core.documents import Document

# === 그래프 구성 ===
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("confirm_retrieval", confirm_retrieval)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite)
graph.add_node("websearch", websearch)
graph.add_node("finalize_response", finalize_response)
graph.add_node("generate_accident_report", generate_accident_report_node)
graph.add_node("grade_report_quality", grade_report_quality)

# === 흐름 정의 ===
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "confirm_retrieval")

graph.add_conditional_edges(
    "confirm_retrieval",
    lambda s: s.get("route", "generate"),
    {"generate": "generate", "rewrite": "rewrite"}
)

graph.add_edge("rewrite", "retrieve")  # rewrite 후 다시 검색

graph.add_edge("websearch", "generate")

graph.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "websearch": "websearch",
        "finalize_response": "finalize_response",
    },
)

graph.add_edge("finalize_response", "generate_accident_report")

graph.add_conditional_edges(
    "generate_accident_report",
    grade_report_quality,
    {"insufficient": "websearch", "adequate": END},
)

# === 그래프 컴파일 ===
app = graph.compile()


# ✅ 단계별 실행 함수
def run_retrieve(query_text: str):
    """1단계: 문서 검색"""
    init_state = {
        "messages": [HumanMessage(content=query_text)],
        "query": query_text,
        "retries": 0,
        "web_fallback": True,
    }
    final_state = app.invoke(init_state, start_at="retrieve", stop_at="confirm_retrieval")
    return final_state


def run_rewrite(prev_state):
    """2단계: 관련 없음 → rewrite → retrieve"""
    state = app.invoke(prev_state, start_at="rewrite", stop_at="confirm_retrieval")
    return state


def run_generate(prev_state):
    """3단계: 관련 있음 → 보고서 생성"""
    final_state = app.invoke(prev_state, start_at="generate_accident_report")
    return final_state
