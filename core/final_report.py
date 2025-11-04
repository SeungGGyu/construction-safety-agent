from core.agentstate import AgentState
from core.llm_utils import call_llm  # ✅ 공통 LLM 호출 유틸 사용


# === 1. 보고서 생성 함수 ===
def generate_accident_report(rag_output: str) -> str:
    """RAG 기반 사고 정보를 입력받아 건설 사고 재발 방지 대책 보고서를 생성"""

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
        # ✅ 공통 유틸 사용 (llm_utils.py 내부에서 모델, URL, 토큰 모두 처리)
        report_text = call_llm(
            [system_message, user_message],
            temperature=0.3,
            top_p=0.9,
            max_tokens=25000
        )
        return report_text
    except Exception as e:
        print(f"⚠️ 보고서 생성 실패: {e}")
        return "보고서 생성 실패"


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

    # 3️⃣ LangGraph state 업데이트
    return {"report": report_text}
