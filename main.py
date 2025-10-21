from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from core.agentstate import AgentState
from core.retriever import retrieve_node
from core.generate import generate
from core.rewrite import rewrite
from core.websearch import websearch
from core.finalize_response import finalize_response
from core.generation_grader import grade_generation
from core.query import query  
from core.kanana import KANANA 
from core.final_report import generate_accident_report_node
from core.report_grader import grade_report_quality
from core.confirm_retrieval import confirm_retrieval
import sys
import logging
from dotenv import load_dotenv
import os


# === 4) 그래프 구성: 노드에는 '함수'를 넣어야 함! ===
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("confirm_retrieval", confirm_retrieval)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite)
graph.add_node("websearch", websearch)
graph.add_node("finalize_response", finalize_response)
graph.add_node("generate_accident_report", generate_accident_report_node) 
graph.add_node("grade_report_quality", grade_report_quality) 


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
        "generate": "generate",               # 환각 → 재생성
        "rewrite": "rewrite",                 # 유용하지 않음 → 질문 리라이트
        "websearch": "websearch",             # 최대 반복 도달 → 웹 보강
        "finalize_response": "finalize_response",  # grounded + useful → 종료
    },
)

# (4) finalize_response 이후 보고서 생성 및 품질평가 연결
graph.add_edge("finalize_response", "generate_accident_report")

graph.add_conditional_edges(
    "generate_accident_report",
    grade_report_quality,  # ✅ 보고서 품질 평가 함수
    {
        "insufficient": "websearch",  # 부족 → 웹검색 후 보강
        "adequate": END               # 충분 → 종료
    },
)

# === 그래프 컴파일 ===
app = graph.compile()

# === 초기 입력 ===
init_question = query[6]  # ✅ 원하는 질의 인덱스 선택
init_state: AgentState = {
    "messages": [HumanMessage(content=init_question)],
    "query": init_question,
    "retries": 0,
    "web_fallback": True,
}

# === 그래프 실행 ===
final_state = app.invoke(init_state)

# === 출력 ===
print("\n=== 🔹 건설 사고 재발 방지 대책 보고서 생성 결과 ===\n")
print(final_state.get("report", "⚠️ 보고서 생성 실패"))