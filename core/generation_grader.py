# core/generation_grader.py
from pydantic import BaseModel, Field, ValidationError
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
import re, json
from core.agentstate import AgentState
from core.kanana import KANANA  # ✅ 내부에서 직접 LLM 로드

SAFE_YES = {"yes","y","예","네","맞음","true"}
SAFE_NO  = {"no","n","아니오","아님","false"}

def _safe_extract_yesno(text: str) -> str:
    try:
        obj = json.loads(text)
        v = str(obj.get("binary_score","")).strip().lower()
    except Exception:
        t = text.strip().lower()
        if re.search(r'\"binary_score\"\s*:\s*\"yes\"', t): return "yes"
        if re.search(r'\"binary_score\"\s*:\s*\"no\"',  t): return "no"
        return "no"
    if v in {"예","네","맞음","true"}: return "yes"
    if v in {"아니오","아님","false"}: return "no"
    return v


def get_hallucination_grader():
    """FACT 기반 사실성 평가"""
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
    return prompt | KANANA | parser


def get_answer_grader():
    """질문-답변 일치도 평가"""
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
    return prompt | KANANA | parser


MAX_RETRIES = 3

def grade_generation(state: AgentState) -> str:
    """
    생성 결과 평가 노드
    - 사실성(Hallucination) 및 질문 적합도(Answer Quality) 기반으로 라우팅 결정
    """
    question = state.get("query") or state["messages"][0].content
    docs = state.get("selected") or state.get("retrieved") or []
    generation = state.get("candidate_answer", state["messages"][-1].content)
    retries = state.get("retries", 0)
    web_fallback = state.get("web_fallback", True)

    hall_chain = get_hallucination_grader()

    try:
        hall_obj = hall_chain.invoke({
            "documents": "\n\n".join(d.page_content for d in docs[:8]),
            "generation": generation
        })
        hall = (hall_obj.binary_score or "").strip().lower()
    except (OutputParserException, ValidationError):
        raw = KANANA.invoke(
            f"FACTS:\n{''.join(d.page_content for d in docs[:4])}\n\nGENERATION:\n{generation}\n"
            "Answer only as JSON: {\"binary_score\":\"yes|no\"}"
        ).content
        hall = _safe_extract_yesno(raw)

    # === 1단계: 사실성 판단 ===
    if hall == "yes":
        ans_chain = get_answer_grader()
        try:
            ans_obj = ans_chain.invoke({"question": question, "generation": generation})
            ans = (ans_obj.binary_score or "").strip().lower()
        except (OutputParserException, ValidationError):
            raw = KANANA.invoke(
                f"QUESTION:\n{question}\n\nANSWER:\n{generation}\n"
                "Answer only as JSON: {\"binary_score\":\"yes|no\"}"
            ).content
            ans = _safe_extract_yesno(raw)

        # === 2단계: 질문 해결 여부 ===
        if ans == "yes":
            return "finalize_response"
        else:
            return "rewrite" if retries < MAX_RETRIES else ("websearch" if web_fallback else "finalize_response")

    # === 3단계: 환각 존재 시 ===
    else:
        return "generate" if retries < MAX_RETRIES else ("websearch" if web_fallback else "finalize_response")