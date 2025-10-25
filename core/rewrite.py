# core/rewrite.py
# 1025 query rewrite 고도화 필요

import re
from langchain_core.messages import HumanMessage
from core.agentstate import AgentState
from core.kanana import KANANA  # ✅ 내부에서 LLM 직접 불러오기 (KANANA 사용)

def _is_korean(t): 
    return re.search(r"[가-힣]", t or "") is not None

def _clean(s): 
    return re.sub(r"\s+", " ", s.strip().strip("'\"`"))

def rewrite(state: AgentState):
    """
    검색 쿼리 재작성 모듈
    (main.py에서 llm 인자를 따로 넘기지 않아도 내부에서 KANANA를 직접 호출)
    """
    print("==== [QUERY REWRITE: 규정 탐색용] ====")
    q = state.get("query") or state["messages"][0].content

    system_prompt = """
당신은 건설 안전 RAG 시스템의 **검색 쿼리 리라이터**입니다.  
당신의 임무는 Agent가 제시한 부적절한 결과의 원인이 검색 쿼리 문제라고 판단될 때,  
보다 효과적으로 관련 규정을 찾을 수 있는 **최적화된 검색 쿼리**를 생성하는 것입니다.  

<instruction>
- 입력과 같은 언어(한국어)를 유지해야 합니다.  
- 출력은 반드시 한 문장으로 작성합니다.  
- 번역은 금지합니다.  
- 질문형 어미(?, ~인가요 등)는 제거합니다.  
- 군더더기를 제거하고 핵심 명사 위주로만 구성합니다.  
</instruction>

<requirements>
1. 쿼리는 검색 효율성을 높여야 합니다.  
2. 불필요한 단어, 중복된 표현을 제거해야 합니다.  
3. 반드시 간결하고 직관적인 형태여야 합니다.  
4. 쿼리를 변형한 **기본 쿼리**와, 규정 검색을 보강하기 위한 **부스팅 쿼리**를 함께 생성해야 합니다.  
</requirements>

<reference_structure>
- 기본 쿼리: 핵심 명사 중심의 간결한 문장  
- 부스팅 쿼리: 기본 쿼리에 “법규, 기준, 지침, 체크리스트, 조항” 키워드 추가  
</reference_structure>

<example>
입력: "철근콘크리트 공사 중 안전난간대 미설치로 인한 추락 위험이 있는지?"  
기본 쿼리: "철근콘크리트 공사 안전난간대 추락 위험"  
부스팅 쿼리: "철근콘크리트 공사 안전난간대 추락 위험 법규 기준 지침 체크리스트 조항"  
</example>
"""

    # ✅ LLM 호출 (KANANA 사용)
    prompt = f"{system_prompt}\n\n입력:\n{q}\n출력:"
    base = _clean(KANANA.invoke([HumanMessage(content=prompt)]).content or q)

    # 부스팅 쿼리 생성
    suffix_ko = " 법규 기준 지침 체크리스트 조항"
    boosted = _clean((base + suffix_ko).rstrip(" ?"))

    # 언어 보정
    if not _is_korean(base):
        base = _clean(q)
    if not _is_korean(boosted):
        boosted = _clean(q + suffix_ko)

    return {
        "messages": state["messages"] + [HumanMessage(content=base)],
        "query": base,
        "query_candidates": [base, boosted],
        "retries": state.get("retries", 0) + 1,
    }