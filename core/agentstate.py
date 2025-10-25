from typing import Annotated, Sequence, TypedDict, Any
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage          # ✅ 최신 구조
from langgraph.graph.message import add_messages
from langchain_core.documents import Document            # ✅ 최신 구조
# from langchain.schema import Document
  # 문서 컨테이너

class AgentState(TypedDict):
    # 1) 대화 이력: 그래프 전체에서 계속 누적
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 2) 질의/의도 파싱 결과 (router나 reformulator가 채움)
    query: NotRequired[str]                 # 최종 검색용 쿼리
    intent: NotRequired[str]                # 예: "lookup", "reason", "summarize"

    # 3) 검색 단계 산출물
    retrieved: NotRequired[list[Document]]  # 원본 검색 결과
    selected: NotRequired[list[Document]]   # 재랭크/필터 후 컨텍스트로 쓸 하위셋
    sources: NotRequired[list[dict[str, Any]]]  # 간단한 출처 요약 (filename, page, idx 등)

    # 4) 생성/검증 단계 산출물
    draft: NotRequired[str]                 # 1차 초안(검증 전)
    answer: NotRequired[str]                # 최종 답변(검증/인용 반영 후)
    citations: NotRequired[list[dict[str, Any]]]  # 인용 메타(문서 id, passage 범위 등)

    # 5) 툴 상호작용/제어
    tool_calls: NotRequired[list[dict[str, Any]]] # 호출 내역/결과 로그
    route: NotRequired[str]                       # 라우터가 선택한 경로 태그
    meta: NotRequired[dict[str, Any]]             # 기타 상태(스텝 카운트, 플래그 등)

    # 루프 제어
    retries: int                 # generate/rewrite 루프 카운트
    web_fallback: bool           # 웹 보강을 시도할지 플래그
    candidate_answer: str        # generate 직후 임시 답변(검증용)

