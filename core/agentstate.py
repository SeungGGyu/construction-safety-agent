from typing import Annotated, Sequence, TypedDict, Any
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain.schema import Document  # 문서 컨테이너

class AgentState(TypedDict):
    """
    전체 LangGraph 실행 동안 공유되는 Agent의 상태 정의
    """

    # 1️⃣ 대화 이력: 그래프 전체에서 계속 누적
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 2️⃣ 질의/의도 파싱 결과
    query: NotRequired[str]                 # 최종 검색용 쿼리
    intent: NotRequired[str]                # 예: "lookup", "reason", "summarize"

    # 3️⃣ 검색 단계 산출물
    retrieved: NotRequired[list[Document]]  # 원본 검색 결과
    selected: NotRequired[list[Document]]   # 재랭크/필터 후 컨텍스트로 쓸 하위셋
    sources: NotRequired[list[dict[str, Any]]]  # 간단한 출처 요약 (filename, page, idx 등)

    # 4️⃣ 생성/검증 단계 산출물
    draft: NotRequired[str]                 # 1차 초안(검증 전)
    answer: NotRequired[str]                # 최종 답변(검증/인용 반영 후)
    candidate_answer: NotRequired[str]      # generate 직후 임시 답변(검증용)
    citations: NotRequired[list[dict[str, Any]]]  # 인용 메타(문서 id, passage 범위 등)

    # 5️⃣ 최종 보고서 단계
    report: NotRequired[str]                # ✅ final_report.py에서 생성된 보고서 텍스트

    # 6️⃣ 툴 상호작용/제어
    tool_calls: NotRequired[list[dict[str, Any]]] # 호출 내역/결과 로그
    route: NotRequired[str]                       # 라우터가 선택한 경로 태그
    meta: NotRequired[dict[str, Any]]             # 기타 상태(스텝 카운트, 플래그 등)

    # 7️⃣ 루프 제어 변수
    retries: int                 # generate/rewrite 루프 카운트
    web_fallback: bool           # 웹 보강을 시도할지 플래그
