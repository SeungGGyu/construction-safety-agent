import re
from langchain_core.messages import HumanMessage, AIMessage
from core.agentstate import AgentState
from core.kanana import KANANA

def _is_korean(t): return re.search(r"[가-힣]", t or "") is not None
def _clean(s): return re.sub(r"\s+", " ", s.strip().strip("'\"`"))

def rewrite(state: AgentState):
    print("==== [QUERY REWRITE: 규정 탐색용] ====")
    q = state.get("query") or state["messages"][0].content
    is_ko = _is_korean(q)

    # 1) LLM으로 군더더기 제거(번역 금지)
    inst_ko = ("입력과 같은 언어로 한 문장만. 번역 금지. 핵심명사만 남기고 군더더기 제거. 물음표 금지.")
    inst_en = ("Use the SAME language as input. One sentence, no question mark, no translation, keep core nouns only.")
    prompt = f"{inst_ko if is_ko else inst_en}\n입력:\n{q}\n출력:"
    base = _clean(KANANA.invoke([HumanMessage(content=prompt)]).content or q)

    # 2) 규정 키워드 부스팅 쿼리 생성
    suffix_ko = " 법규 기준 지침 체크리스트 조항"
    suffix_en = " regulation standard guideline checklist clause"
    boosted = _clean((base + (suffix_ko if is_ko else suffix_en)).rstrip(" ?"))

    # 언어 어긋나면 원문 유지
    if is_ko and not _is_korean(base): base = _clean(q)
    if is_ko and not _is_korean(boosted): boosted = _clean(q + suffix_ko)

    return {
        "messages": state["messages"] + [HumanMessage(content=base)],
        "query": base,                           # 기본 쿼리로 1차 시도
        "query_candidates": [base, boosted],     # retrieve에서 둘 다 사용
    }

