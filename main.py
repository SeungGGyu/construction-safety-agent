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
import sys
import logging
from dotenv import load_dotenv
import os


# === 4) 그래프 구성: 노드에는 '함수'를 넣어야 함! ===
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite)
graph.add_node("websearch", websearch)
graph.add_node("finalize_response", finalize_response)
graph.add_node("generate_accident_report", generate_accident_report_node) 
graph.add_node("grade_report_quality", grade_report_quality) 


graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
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

# graph.add_edge("finalize_response", END)
# ✅ finalize_response 이후 보고서 생성으로 연결


# 보고서 생성 및 품질 평가 ===
graph.add_edge("finalize_response", "generate_accident_report")

graph.add_conditional_edges(
    "generate_accident_report",
    grade_report_quality,  # ✅ 보고서 충분성 평가
    {
        "insufficient": "websearch",  # 부족하면 다시 웹 검색 후 재작성
        "adequate": END               # 충분하면 종료
    },
)


app = graph.compile()


init_question = query[0]  # 네가 만든 query 리스트에서 하나 선택

init_state: AgentState = {
    "messages": [HumanMessage(content=init_question)],
    "query": init_question,
    "retries": 0,          # 새로 추가한 필드
    "web_fallback": True,  # 웹 보강 허용 여부
}

final_state = app.invoke(init_state)

# print("\n=== 최종 응답 ===\n")
# print(final_state["messages"][-1].content)


print("\n=== 🔹 건설 사고 재발 방지 대책 보고서 생성 결과 ===\n")
print(final_state.get("report", "⚠️ 보고서 생성 실패"))
