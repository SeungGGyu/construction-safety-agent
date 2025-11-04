# core/confirm_retrieval.py
import re
from core.agentstate import AgentState
from langchain.schema import Document
from bs4 import BeautifulSoup

def _clean_html(text: str) -> str:
    """HTML 태그 제거 및 줄바꿈 유지"""
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return _prettify_text(text)

def _prettify_text(text: str) -> str:
    """표·기호 구조가 깨진 텍스트를 사람이 읽기 좋게 재정렬"""
    text = re.sub(r"[\u2027•․·]+", "·", text)        # 중간점 통일
    text = re.sub(r"\s+", " ", text)                 # 과도한 공백 제거
    text = re.sub(r"(\.)([가-힣])", r"\1\n\2", text) # 문장 구분시 줄바꿈 추가
    text = re.sub(r"(·\s*)", r"\n- ", text)          # · 기호를 리스트 형식으로 변환
    text = re.sub(r"([가-힣])(\s*:\s*)", r"\1\n", text)
    text = text.strip()
    return text

def confirm_retrieval(state: AgentState):
    """
    Human-in-the-loop 확인 단계 (CLI 버전)
    - 검색 결과(청킹 데이터)를 사람이 검토하고 필요 시 제외할 수 있음
    """
    docs = state.get("retrieved", [])
    if not docs:
        print("\n  검색된 문서가 없습니다. 쿼리를 재작성합니다.")
        return {"route": "rewrite"}

    print("\n🔍 === 검색 결과 미리보기 ===")

    # === 모든 검색 문서 표시 ===
    for i, doc in enumerate(docs):
        meta = doc.metadata
        file = meta.get("source")
        section = meta.get("section")

        clean_text = _clean_html(doc.page_content.strip())

        print(f"\n📄 [{i+1}] 문서 정보")
        print(f"   ┣ 파일명: {file}")
        print(f"   ┣ 섹션: {section}")
        print(f"   ┗ 내용:\n{clean_text}")
        print("-" * 120)

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
            excluded_indices = [
                int(x.strip()) - 1
                for x in exclude_input.split(",")
                if x.strip().isdigit() and 1 <= int(x.strip()) <= max_idx
            ]
            display_nums = [i + 1 for i in excluded_indices]
            print(f"🚫 제외 문서 번호: {display_nums}")
        except Exception:
            print("⚠️ 제외 번호 입력을 이해할 수 없습니다. 모든 문서를 유지합니다.")
            excluded_indices = []

    # === 최종 선택 문서 반영 ===
    selected_docs = [d for i, d in enumerate(docs) if i not in excluded_indices]
    print(f"\n✅ {len(selected_docs)}개 문서를 유지하고 다음 단계로 진행합니다.")

    return {
        "retrieved": selected_docs,
        "selected": selected_docs,
        "docs_text": "\n\n".join(
            f"[{i+1}] {_clean_html(d.page_content)}" for i, d in enumerate(selected_docs)
        ),
        "sources": [
            {
                "idx": i + 1,
                "file": d.metadata.get("file", "?"),
                "section": d.metadata.get("section", "?"),
            }
            for i, d in enumerate(selected_docs)
        ],
        "route": "generate",
    }
