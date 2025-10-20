# core/websearch.py
from core.agentstate import AgentState
from langchain_tavily import TavilySearch
from langchain.schema import Document
import os

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)
if not TAVILY_API_KEY:
    raise ValueError("❌ Tavily API 키가 없습니다. 환경 변수 TAVILY_API_KEY를 설정하세요.")

tavily = TavilySearch(tavily_api_key=TAVILY_API_KEY, k=5)

def websearch(state: AgentState):
    """웹 보강 검색 (사용자 직접 query 입력형)"""
    print("\n🌍 [WebSearch] 보강 검색을 수행합니다.")
    print("현재 질문:", state.get("query") or state["messages"][-1].content)
    user_query = input("\n검색할 쿼리를 직접 입력하세요 (예: '철근콘크리트 안전난간대 설치 기준 KOSHA GUIDE'):\n> ").strip()

    if not user_query:
        print("⚠️ 입력이 비어 있습니다. 기존 쿼리를 사용합니다.")
        user_query = state.get("query") or state["messages"][-1].content

    print(f"\n🔎 [Tavily] '{user_query}' 로 웹 검색 중...")

    # ✅ Tavily 실행 (딕셔너리 형태 반환)
    response = tavily.invoke({"query": user_query + " site:kosha.or.kr OR site:moel.go.kr OR site:kosha.net"})

    # ✅ 실제 결과만 추출
    raw_results = response.get("results", [])
    docs_web = [
        Document(
            page_content=item.get("content", ""),
            metadata={
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "source": "TavilyWeb"
            }
        )
        for item in raw_results
        if item.get("content")
    ]

    # ✅ 기존 문서와 병합
    prev_docs = state.get("retrieved") or []
    new_docs = prev_docs + docs_web

    print(f"✅ {len(docs_web)}개 웹 문서 보강 완료.\n")
    for i, d in enumerate(docs_web):
        title = d.metadata.get("title", "No Title")
        snippet = d.page_content[:200].replace("\n", " ")
        url = d.metadata.get("url", "")
        print(f"[{i+1}] {title}\n{snippet}...\nURL: {url}\n")

    return {
        "retrieved": new_docs,
        "selected": new_docs,
        "web_fallback": False
    }
