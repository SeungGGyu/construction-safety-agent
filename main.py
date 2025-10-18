# main.py
import argparse
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
from core.final_report import generate_accident_report

# 모델 불러오기

# === argparse 추가 ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="kanana", choices=["kanana", "qwen"])
args = parser.parse_args()

if args.model == "kanana":
    from core.kanana import KANANA
    LLM = KANANA
    print("🚀 Using KANANA model")
else:
    from core.qwen import QWEN
    LLM = QWEN
    print("🚀 Using QWEN model")

# === 4) 그래프 구성 ===
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", lambda state: generate(state, llm=LLM))  # ← 모델 주입
graph.add_node("rewrite", lambda state: rewrite(state, llm=LLM))
graph.add_node("websearch", websearch)
graph.add_node("finalize_response", finalize_response)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("rewrite", "retrieve")
graph.add_edge("websearch", "generate")
graph.add_conditional_edges(
    "generate",
    lambda state: grade_generation(state, llm=LLM),  
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "websearch": "websearch",
        "finalize_response": "finalize_response",
    },
)

graph.add_edge("finalize_response", END)

app = graph.compile()

# 초기 state
init_question = query[6]
init_state: AgentState = {
    "messages": [HumanMessage(content=init_question)],
    "query": init_question,
    "retries": 0,
    "web_fallback": True,
}

final_state = app.invoke(init_state)

rag_output = final_state["messages"][-1].content
print("\n=== 최종 응답 ===\n")
print(rag_output)

# 보고서 생성기로 전달
report = generate_accident_report(rag_output)
print("\n===== 건설 사고 재발 방지 대책 보고서 초안 =====\n")
print(report)
