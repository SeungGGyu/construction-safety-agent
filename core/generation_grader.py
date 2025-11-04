# core/generation_grader.py
import re, json
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from core.agentstate import AgentState
from core.llm_utils import call_llm  # ✅ 공용 LLM 호출 유틸 사용


SAFE_YES = {"yes", "y", "예", "네", "맞음", "true"}
SAFE_NO = {"no", "n", "아니오", "아님", "false"}


# === ✅ 안전한 yes/no 판별 ===
def _safe_extract_yesno(text: str) -> str:
    """LLM 응답에서 binary_score를 안전하게 추출"""
    try:
        obj = json.loads(text)
        v = str(obj.get("binary_score", "")).strip().lower()
    except Exception:
        t = text.strip().lower()
        if re.search(r'\"binary_score\"\s*:\s*\"yes\"', t):
            return "yes"
        if re.search(r'\"binary_score\"\s*:\s*\"no\"', t):
            return "no"
        return "no"

    if v in SAFE_YES:
        return "yes"
    if v in SAFE_NO:
        return "no"
    return v


# === ✅ FACT 기반 사실성 평가 ===
def get_hallucination_grader():
    """FACT 기반 사실성 평가용 프롬프트"""
    class GradeHallucinations(BaseModel):
        binary_score: str = Field(description="'yes'이면 FACTS에 근거, 'no'이면 근거 없음")

    parser = PydanticOutputParser(pydantic_object=GradeHallucinations)
    prompt = PromptTemplate(
        template=(
            "당신은 생성문이 주어진 FACTS에 근거하는지 판정하는 채점기입니다.\n"
            "규칙:\n"
            "1) 반드시 JSON만 출력하세요. (설명, 주석, 코드블록 금지)\n"
            "2) 값은 'yes' 또는 'no' 중 하나여야 합니다.\n"
            "3) 근거 불충분, 모호할 경우 'no'.\n\n"
            "FACTS:\n```facts\n{documents}\n```\n\n"
            "GENERATION:\n```gen\n{generation}\n```\n\n"
            "JSON 스키마:\n{format_instructions}\n"
        ),
        input_variables=["documents", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser


# === ✅ 질문-답변 일치도 평가 ===
def get_answer_grader():
    """질문-답변 일치도 평가용 프롬프트"""
    class GradeAnswer(BaseModel):
        binary_score: str = Field(description="'yes'이면 답변이 질문을 해결, 'no'이면 해결 못함")

    parser = PydanticOutputParser(pydantic_object=GradeAnswer)
    prompt = PromptTemplate(
        template=(
            "당신은 주어진 ANSWER가 QUESTION을 실제로 해결하는지 판정하는 채점기입니다.\n"
            "규칙:\n"
            "1) 반드시 JSON만 출력하세요. (설명, 코드블록 금지)\n"
            "2) 값은 'yes' 또는 'no' 중 하나여야 합니다.\n\n"
            "QUESTION:\n```q\n{question}\n```\n\n"
            "ANSWER:\n```a\n{generation}\n```\n\n"
            "JSON 스키마:\n{format_instructions}\n"
        ),
        input_variables=["question", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser


# === ✅ 메인 평가 함수 ===
MAX_RETRIES = 3

def grade_generation(state: AgentState) -> str:
    """생성 결과의 사실성 및 질문 해결 여부를 단계적으로 평가"""
    question = state.get("query") or state["messages"][0].content
    docs = state.get("selected") or state.get("retrieved") or []
    generation = state.get("candidate_answer", state["messages"][-1].content)
    retries = state.get("retries", 0)
    web_fallback = state.get("web_fallback", True)

    # === 1️⃣ 사실성 평가 (FACTS 기반) ===
    prompt, parser = get_hallucination_grader()
    filled_prompt = prompt.format(
        documents="\n\n".join(d.page_content for d in docs[:8]),
        generation=generation
    )

    raw = call_llm([
        {"role": "system", "content": "당신은 FACTS 기반 채점기입니다. 오직 JSON만 출력하세요."},
        {"role": "user", "content": filled_prompt}
    ])
    hall = _safe_extract_yesno(raw)

    if hall == "yes":
        # === 2️⃣ 질문 해결 여부 평가 ===
        prompt, parser = get_answer_grader()
        filled_prompt = prompt.format(question=question, generation=generation)
        raw = call_llm([
            {"role": "system", "content": "당신은 답변이 질문을 해결하는지 판단하는 평가자입니다. 반드시 JSON만 출력하세요."},
            {"role": "user", "content": filled_prompt}
        ])
        ans = _safe_extract_yesno(raw)

        # === 결과 라우팅 ===
        if ans == "yes":
            return "finalize_response"
        else:
            return "rewrite" if retries < MAX_RETRIES else ("websearch" if web_fallback else "finalize_response")

    else:
        return "generate" if retries < MAX_RETRIES else ("websearch" if web_fallback else "finalize_response")
