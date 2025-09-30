from typing import Literal
import re, json
from pydantic import BaseModel, Field, ValidationError
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from core.agentstate import AgentState
# --- helpers ---
def _is_korean(text: str) -> bool:
    return re.search(r"[가-힣]", text or "") is not None

class Grade(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, else 'no'")

def _rel_prompt(parser: PydanticOutputParser, ko: bool) -> PromptTemplate:
    if ko:
        tmpl = (
            "당신은 사용자의 질문과 검색된 문서의 관련성을 평가하는 채점기입니다.\n"
            "- 키워드 일치 또는 의미적 유사성이 있으면 관련입니다.\n"
            "- 오직 JSON만 출력하세요. 설명/코드블록 금지.\n"
            "- binary_score 값은 반드시 소문자 'yes' 또는 'no' 여야 합니다.\n\n"
            "DOCUMENT:\n{context}\n\n"
            "QUESTION:\n{question}\n\n"
            "JSON 스키마:\n{format_instructions}\n"
        )
    else:
        tmpl = (
            "You are a grader assessing the relevance of a retrieved document to a user question.\n"
            "- Keyword match OR semantic similarity counts as relevant.\n"
            "- Respond ONLY with JSON. No explanations or code fences.\n"
            "- The field 'binary_score' must be lowercase 'yes' or 'no'.\n\n"
            "DOCUMENT:\n{context}\n\n"
            "QUESTION:\n{question}\n\n"
            "Respond with a compact JSON object that matches this schema:\n"
            "{format_instructions}\n"
        )
    return PromptTemplate(
        template=tmpl,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

# --- improved grader ---
def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    # 1) 후보 문서 모으기 (상위 N개)
    docs = (state.get("retrieved") or state.get("selected") or [])[:8]
    if not docs:
        return "rewrite"

    question = state.get("query") or state["messages"][0].content
    parser = PydanticOutputParser(pydantic_object=Grade)
    prompt = _rel_prompt(parser, ko=_is_korean(question + (docs[0].page_content if docs else "")))
    chain = prompt | KANANA | parser

    yes_votes = 0
    selected = []
    weighted_yes = 0.0

    def _rerank_score(d):
        try:
            return float(d.metadata.get("rerank_score", 0.5))
        except Exception:
            return 0.5

    for d in docs:
        # 2) 각 문서를 개별 채점 (토큰 보호를 위해 앞부분만)
        ctx = d.page_content[:1800]
        try:
            res: Grade = chain.invoke({"question": question, "context": ctx})
            label = (res.binary_score or "").strip().lower()
        except (OutputParserException, ValidationError):
            # LLM이 JSON을 어기면 약식 추출
            raw = (prompt | KANANA).invoke({"question": question, "context": ctx})
            text = getattr(raw, "content", str(raw)).lower()
            label = "yes" if ("\"binary_score\":\"yes\"" in text or re.search(r"\byes\b", text)) else "no"

        if label == "yes":
            yes_votes += 1
            selected.append(d)
            weighted_yes += _rerank_score(d)

    # 3) 선택 문서/컨텍스트를 state에 반영
    state["selected"] = selected
    state["docs_text"] = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(selected))

    # 4) 분기 기준: 투표 비율 + 재랭커 가중 평균
    total = max(1, len(docs))
    vote_ratio = yes_votes / total
    weighted_mean = (weighted_yes / max(1, yes_votes)) if yes_votes else 0.0

    return "generate" if (yes_votes >= 1 and (vote_ratio >= 0.5 or weighted_mean >= 0.55)) else "rewrite"

