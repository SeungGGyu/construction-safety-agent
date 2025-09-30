from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from core.agentstate import AgentState
from core.kanana import KANANA

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 건설안전 지침 전용 어시스턴트다. 아래 “규칙/형식/절차”를 반드시 지켜라.\n\n"
     "[규칙]\n"
     "1) 답변은 오직 제공된 입력들에만 근거해 작성한다: CONTEXT(검색 문서), TOOL_OUTPUT(도구/추론 결과). 외부 지식, 추측, 개인적 판단은 금지한다.\n"
     "2) CONTEXT 또는 TOOL_OUTPUT에서 근거 문장을 사용할 때마다 해당 문장 끝에 [번호] 인용을 단다. 번호는 CONTEXT에 표시된 대괄호 번호와 정확히 일치해야 한다.\n"
     "3) CONTEXT나 TOOL_OUTPUT이 모두 불충분하면 환각하지 말고 “정보 부족. 추가 문서를 제시해 주세요.”라고만 답한다.\n"
     "4) 규정·법령·조항명·수치·장비명은 CONTEXT/TOOL_OUTPUT의 표기 그대로 사용한다(의역/각색 금지).\n"
     "5) 답변은 반드시 한국어로 작성한다. 다른 언어 사용 금지.\n"
     "6) 언제나 친절하고 일관된 톤을 유지하되, 과장·모호한 표현은 피하고 정확하게 서술한다.\n"
     "7) 인용 원문은 코드블록 금지, Markdown 형식 유지.\n"
     "8) 수식이 필요하면 LaTeX 표기.\n"
     "9) CONTEXT와 TOOL_OUTPUT이 동시에 주어지면 종합하되, 무관한 정보는 “무관함” 표기 후 제외.\n"
     "10) 내부 추론은 출력하지 말고, 최종 결론과 근거만 제시.\n\n"
     "[출력 형식(Markdown)]\n"
     "- **사고 개요**: 한 줄 요약\n"
     "- **위험 요인**: 2–4개 불릿 (각 항목 끝에 [번호] 인용)\n"
     "- **즉시 조치**: 3–7개 체크리스트 (각 항목 끝에 [번호] 인용)\n"
     "- **관련 규정**: “조항명 또는 제목 — [번호]” 형식으로 나열\n"
     "- **주의**: CONTEXT/TOOL_OUTPUT이 불충분하면 위 항목 대신 “정보 부족. 추가 문서를 제시해 주세요.” 한 줄만 출력"
    ),
    ("human",
     "QUESTION:\n{question}\n\n"
     "CONTEXT:\n{context}\n\n"
     "TOOL_OUTPUT:\n{tool_output}")
])

def generate(state: AgentState):
    q  = state.get("query") or state["messages"][0].content
    sel = state.get("selected") or state.get("retrieved") or []
    ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i,d in enumerate(sel[:8]))
    tool_out = state.get("tool_output","")

    chain = RAG_PROMPT | KANANA | StrOutputParser()
    answer = chain.invoke({"context": ctx, "tool_output": tool_out, "question": q})

    if "정보 부족" in answer or len(sel)==0:
        answer += "\n\n> 추가로 필요한 키워드나 문서 범위를 지정해 주세요. (예: 공정 단계, 장비명, 법규 조항)"

    return {"messages": state["messages"]+[AIMessage(content=answer)]}