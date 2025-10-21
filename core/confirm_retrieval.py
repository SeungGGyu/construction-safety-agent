# core/confirm_retrieval.py
from core.agentstate import AgentState
from langchain.schema import Document

def confirm_retrieval(state: AgentState):
    """
    Human-in-the-loop 확인 단계 (CLI 버전)
    터미널에서 검색 결과를 보여주고, 사용자에게 yes/no 및 제외 문서를 입력받음.
    """
    docs = state.get("retrieved", [])
    if not docs:
        print("\n⚠️ 검색된 문서가 없습니다. 쿼리를 재작성합니다.")
        return {"route": "rewrite"}

    print("\n🔍 === 검색 결과 미리보기 ===")
    for i, doc in enumerate(docs):
        filename = doc.metadata.get("filename", "?")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content.strip().replace("\n", " ")[:300]
        print(f"\n[{i+1}] ({filename} p.{page})\n{preview}...")

    # === yes/no 입력 ===
    while True:
        user_input = input("\n이 문서들이 질문과 관련이 있나요? (yes/no): ").strip().lower()
        if user_input in {"yes", "y", "예", "네"}:
            break
        elif user_input in {"no", "n", "아니오", "아님"}:
            print("🔄 사용자가 검색 결과를 거부했습니다. 쿼리를 재작성합니다.")
            return {"route": "rewrite"}
        else:
            print("❗ 'yes' 또는 'no'로 입력해주세요.")

    # === 제외 문서 선택 ===
    exclude_input = input("\n제외할 문서 번호를 입력하세요 (쉼표 구분, 없으면 Enter): ").strip()
    excluded_indices = []
    if exclude_input:
        try:
            max_idx = len(docs)
            # 1~N → 0~N-1 변환, 범위 확인
            excluded_indices = [
                int(x.strip()) - 1
                for x in exclude_input.split(",")
                if x.strip().isdigit() and 1 <= int(x.strip()) <= max_idx
            ]

            # ✅ 표시할 때는 항상 사용자 기준 번호(+1)
            display_nums = [i + 1 for i in excluded_indices]
            print(f"🚫 제외 문서 번호: {display_nums}")

        except Exception:
            print("⚠️ 제외 번호 입력을 이해할 수 없습니다. 모든 문서를 유지합니다.")
            excluded_indices = []

    # === 최종 선택 문서 반영 ===
    selected_docs = [d for i, d in enumerate(docs) if i not in excluded_indices]
    print(f"\n✅ {len(selected_docs)}개 문서를 유지하고 다음 단계로 진행합니다.")

    return {
    # ✅ 덮어쓰기: retrieved, selected, docs_text 모두 새 리스트로 대체
    "retrieved": selected_docs,
    "selected": selected_docs,
    "docs_text": "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(selected_docs)),

    # ✅ sources도 새로 만들어줌 (generate에 그대로 전달됨)
    "sources": [
        {
            "idx": i + 1,
            "filename": d.metadata.get("filename", "?"),
            "page": d.metadata.get("page", "?"),
        }
        for i, d in enumerate(selected_docs)
    ],

    # ✅ route
    "route": "generate",
}