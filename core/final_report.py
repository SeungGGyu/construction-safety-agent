from core.agentstate import AgentState
from core.llm_utils import call_llm
from langchain.schema import AIMessage # ✅ 공통 LLM 호출 유틸 사용
import traceback
import json


# === 1. 보고서 생성 함수 ===
def generate_accident_report(rag_output: str) -> str:
    """
    RAG 기반 사고 정보를 입력받아 건설 사고 재발 방지 대책 보고서를 생성
    """
    system_message = {
        "role": "system",
        "content": """
당신은 건설 안전 및 사고 재발 방지 보고서를 전문적으로 작성하는 전문가입니다.  
입력으로 제공되는 RAG 분석 결과(<chunk>)에는 ‘사고 개요’, ‘위험 요인’, ‘즉시 조치’, ‘관련 규정’이 포함되어 있습니다.  
이 정보를 바탕으로 **Word 기준 약 4페이지 분량(약 1800~2200 단어)**의 정식 보고서를 작성하십시오.

--- 중략 (prompt는 그대로 유지) ---
"""
    }

    user_message = {
        "role": "user",
        "content": f"다음은 RAG 분석 결과이다. 이를 토대로 보고서를 작성하라:\n\n{rag_output}"
    }

    try:
        print("🧠 [LLM 호출 시작] 보고서 생성 요청 중...")
        report_text = call_llm(
            [system_message, user_message],
            temperature=0.3,
            top_p=0.9,
            max_tokens=25000
        )

        if not report_text or "⚠️" in report_text:
            print("⚠️ LLM 응답 비정상 또는 실패:", report_text)
            return "보고서 생성 실패 (LLM 응답 없음 또는 오류)"

        print("✅ 보고서 생성 완료")
        return report_text

    except Exception as e:
        print("❌ 보고서 생성 중 예외 발생!")
        print(f"예외 타입: {type(e).__name__}")
        print(f"예외 메시지: {e}")
        print(traceback.format_exc())

        # 혹시 response.text가 JSON 파싱 실패할 경우 확인
        try:
            print("응답 디버그 정보:", json.dumps(report_text, ensure_ascii=False)[:300])
        except Exception:
            pass

        return "보고서 생성 실패 (예외 발생)"


# === 2. LangGraph 연동용 노드 함수 ===
def generate_accident_report_node(state: AgentState):
    """
    LangGraph에서 호출되는 보고서 생성 노드.
    RAG 최종 결과를 받아 generate_accident_report() 함수를 실행하고,
    결과를 state에 저장한다.
    """
    # 1️⃣ RAG 결과 가져오기
    rag_output = state["messages"][-1].content

    # 2️⃣ 보고서 생성
    report_text = generate_accident_report(rag_output)

    # 3️⃣ LangGraph state에 AI 메시지 추가 (✅ 핵심 수정)
    state["messages"].append(AIMessage(content=report_text))

    # 4️⃣ report 키에도 저장 (선택적, 이후 노드 접근용)
    state["report"] = report_text

    # 5️⃣ 전체 state 반환
    return state
