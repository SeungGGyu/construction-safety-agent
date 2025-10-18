from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from core.agentstate import AgentState

# === 프롬프트 정의 (관련 규정은 제외) ===
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """
당신은 건설 안전 지침 전용 어시스턴트입니다.  
당신의 임무는 제공된 CONTEXT(검색 문서)와 TOOL_OUTPUT(도구/추론 결과)를 기반으로 
**사고 개요, 위험 요인, 즉시 조치**를 작성하는 것입니다.  

<instruction>
- 반드시 CONTEXT와 TOOL_OUTPUT만을 근거로 사용하며, 외부 지식·추측·개인적 판단은 금지됩니다.  
- 답변은 항상 한국어로 작성해야 합니다.  
- **사고 개요, 위험 요인, 즉시 조치**의 각 항목 끝에는 반드시 [#n] 또는 [#n, #m] 형태의 인용 번호를 붙여야 합니다.  
- **관련 규정**은 작성하지 마세요. 관련 규정은 코드에서 SOURCES를 기반으로 자동 생성됩니다.  
</instruction>

<requirements>
1. 출력은 명확하고 계층적인 구조를 가져야 합니다.  
2. **사고 개요, 위험 요인, 즉시 조치** 항목은 반드시 포함해야 합니다.  
3. 인용 번호는 retriever가 반환한 문서 번호 [#n] 형식과 일치해야 합니다.  
4. 항상 전문적이고 형식적인 문체를 유지하세요.  
</requirements>

<reference_structure>
- **사고 개요**: 한 줄 요약 [#n]  
- **위험 요인**: 2–4개 불릿 (각 항목 끝에 [#n])  
- **즉시 조치**: 3–7개 체크리스트 (각 항목 끝에 [#n])  
</reference_structure>
"""),
    ("human",
     "QUESTION:\n{question}\n\n"
     "CONTEXT:\n{context}\n\n"
     "TOOL_OUTPUT:\n{tool_output}")
])

# === 보고서 생성 함수 ===
def generate(state: AgentState, llm):
    """
    건설 사고 보고서 1차 생성 노드
    - llm: KANANA 또는 QWEN (main.py에서 주입)
    """
    q = state.get("query") or state["messages"][0].content
    sel = state.get("selected") or state.get("retrieved") or []

    # CONTEXT: 본문 텍스트 (최대 8개)
    ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(sel[:8]))
    tool_out = state.get("tool_output", "")

    # LLM으로 사고 개요/위험 요인/즉시 조치 생성
    chain = RAG_PROMPT | llm | StrOutputParser()
    body_answer = chain.invoke({
        "context": ctx,
        "tool_output": tool_out,
        "question": q,
    })

    # SOURCES: retriever.py에서 만들어둔 sources 활용 → 관련 규정은 코드에서 직접 생성
    src_list = state.get("sources", [])
    if src_list:
        related = "\n".join(
            f"- {s['filename']}, p.{s['page']} [#{s['idx']}]"
            for s in src_list
        )
        related_section = f"\n\n**관련 규정**:\n{related}"
    else:
        related_section = "\n\n**관련 규정**:\n정보 부족. 추가 문서를 제시해 주세요."

    # 최종 답변 합치기
    answer = body_answer + related_section

    if "정보 부족" in answer or len(sel) == 0:
        answer += "\n\n> 추가로 필요한 키워드나 문서 범위를 지정해 주세요. (예: 공정 단계, 장비명, 법규 조항)"

    return {
        "messages": state["messages"] + [AIMessage(content=answer)],
        "candidate_answer": answer,                # grade_generation 평가용
        "retries": state.get("retries", 0) + 1,    # 루프 카운트 증가
    }
