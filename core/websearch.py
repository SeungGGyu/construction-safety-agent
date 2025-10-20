# core/websearch.py
from core.agentstate import AgentState
from core.retriever import retriever_instance   # ✅ 추가
import os
from langchain_community.retrievers import TavilySearchAPIRetriever

# 이전 버전 
# def websearch(state: AgentState):
#     # 여기서는 placeholder: 기존 retriever로 다른 인덱스 or 확장 쿼리 재검색
#     q = state.get("query") or state["messages"][-1].content
#     docs_web = retriever_instance.retrieve(q + " 안전기준 법규 조항 체크리스트")
#     # 웹 보강 이후에는 더 이상 웹으로 빠지지 않게 플래그 OFF
#     new_docs = (state.get("retrieved") or []) + docs_web
#     return {
#         "retrieved": new_docs,
#         "selected": new_docs,
#         "web_fallback": False
#     }

def websearch(state: AgentState) -> AgentState:
    """
    Tavily API를 이용한 웹 검색 노드
    - 기존 query 또는 마지막 메시지 내용을 기반으로 웹 검색 수행
    - TavilySearchAPIRetriever로 관련 웹 문서를 가져옴
    - 기존 retrieved 문서와 병합
    """

    os.environ["TAVILY_API_KEY"] = "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"

    # 2️⃣ 검색 쿼리 결정
    query_text = state.get("query") or state["messages"][-1].content
    expanded_query = query_text + " 관련 법규 및 안전 기준"

    #  Tavily retriever 초기화
    tavily_retriever = TavilySearchAPIRetriever(
        api_key=os.environ["TAVILY_API_KEY"],
        k=5,  # 가져올 문서 개수
        search_depth="advanced",  # 기본보다 깊게 검색 (선택)
    )

    # Tavily 웹 검색 수행
    docs_web = tavily_retriever.get_relevant_documents(expanded_query)

    # 기존 retrieved 문서와 병합
    prev_docs = state.get("retrieved", [])
    merged_docs = prev_docs + docs_web

    #  상태 업데이트
    state["retrieved"] = merged_docs
    state["selected"] = merged_docs
    state["web_fallback"] = False  # 웹 보강 이후 추가 fallback 방지

    #  메시지 로그 추가 (선택)
    state["messages"].append(
        {
            "role": "system",
            "content": f"Tavily 검색 결과 {len(docs_web)}건 추가됨.",
        }
    )

    return state