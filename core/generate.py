# core/generate.py
import re
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from core.agentstate import AgentState


# === Qwen API 설정 ===
LLM_URL = "http://211.47.56.73:8908"  # ✅ OpenAI 호환 API 엔드포인트
LLM_TOKEN = "token-abc123"
LLM_MODEL = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"


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
- ⚠️ 인용 번호는 반드시 CONTEXT 내 존재하는 번호만 사용하십시오.  
  예: CONTEXT에 [#1], [#2], [#3], [#4], [#5]만 있다면 이 중에서만 선택해야 합니다.  
  존재하지 않는 번호([#6], [#7], [#8] 등)는 절대 사용하지 마십시오.  
- **관련 규정**은 작성하지 마세요. 관련 규정은 코드에서 SOURCES를 기반으로 자동 생성됩니다.  
</instruction>

<requirements>
1. 출력은 명확하고 계층적인 구조를 가져야 합니다.  
2. **사고 개요, 위험 요인, 즉시 조치** 항목은 반드시 포함해야 합니다.  
3. 인용 번호는 retriever가 반환한 문서 번호 [#n] 형식과 일치해야 합니다.  
4. 항상 전문적이고 형식적인 문체를 유지하세요.  
</requirements>

<output_format>
출력은 반드시 아래 형식을 따르되, 각 항목 개수는 상황에 따라 2~5개로 자유롭게 작성하시오.

**사고 개요**:
- (내용 한 줄 요약) [#n]

**위험 요인**:
- (항목 1) [#n]
- (항목 2) [#n]
- (필요 시 추가 가능)

**즉시 조치**:
- (항목 1) [#n]
- (항목 2) [#n]
- (필요 시 추가 가능)
</output_format>
"""),
    ("human",
     "QUESTION:\n{question}\n\n"
     "CONTEXT:\n{context}\n\n"
     "TOOL_OUTPUT:\n{tool_output}")
])



# === Qwen API 호출 ===
def call_qwen(prompt: str) -> str:
    """로컬 Qwen 서버 호출 (OpenAI 호환 API 방식)"""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "당신은 건설 안전 지침 전용 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4000,
        "temperature": 0.3,
        "top_p": 0.9,
    }
    headers = {
        "Authorization": f"Bearer {LLM_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{LLM_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ LLM 호출 실패: {e}")
        if 'response' in locals():
            print(f"서버 응답: {response.text}")
        return "정보 부족. 추가 문서를 제시해 주세요."


# === 출력 포맷 보정 함수 ===
def format_sections(text: str) -> str:
    """사고 개요 / 위험 요인 / 즉시 조치 구분이 명확하게 되도록 포맷 보정"""
    text = re.sub(r'(사고\s*개요\s*[:：])', r'\n\n**\1** ', text)
    text = re.sub(r'(위험\s*요인\s*[:：])', r'\n\n**\1** ', text)
    text = re.sub(r'(즉시\s*조치\s*[:：])', r'\n\n**\1** ', text)
    return text.strip()


# === 보고서 생성 ===
def generate(state: AgentState):
    """RAG 기반 사고 개요 / 위험 요인 / 즉시 조치 생성"""
    q = state.get("query") or state["messages"][0].content
    sel = state.get("selected") or state.get("retrieved") or []

    # ✅ retrieved 동기화
    if len(state.get("retrieved", [])) != len(sel):
        state["retrieved"] = sel

    # ✅ sources 재생성
    src_list = [
        {
            "idx": i + 1,
            "filename": d.metadata.get("filename", "?"),
            "page": d.metadata.get("page", "?")
        }
        for i, d in enumerate(sel)
    ]
    state["sources"] = src_list

    # ✅ CONTEXT 구성
    ctx = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(sel[:8]))
    tool_out = state.get("tool_output", "")

    # ✅ 사용할 인용 번호 명시
    valid_refs = ", ".join(f"#{s['idx']}" for s in src_list)
    prompt_text = RAG_PROMPT.format(context=ctx, tool_output=tool_out, question=q)
    prompt_text += f"\n\n⚠️ 사용할 수 있는 인용 번호는 다음과 같습니다: {valid_refs}"

    # ✅ Qwen 호출
    body_answer = call_qwen(prompt_text)
    body_answer = format_sections(body_answer)

    # ✅ 존재하지 않는 인용번호 제거 ([#6] 이상 등)
    max_idx = len(src_list)
    invalid_refs = re.findall(r"\[\s*#\s*(\d+)\s*\]", body_answer)
    for ref in set(invalid_refs):
        try:
            if int(ref) > max_idx:
                body_answer = re.sub(rf"\[\s*#\s*{ref}\s*\]", "", body_answer)
        except Exception:
            pass

    # ✅ 관련 규정 섹션 (sources 기반)
    if src_list:
        related = "\n".join(
            f"- {s['filename']}, p.{s['page']} [#{s['idx']}]"
            for s in src_list if s["filename"] != "?"
        )
        related_section = f"\n\n**관련 규정**:\n{related}" if related else "\n\n**관련 규정**:\n정보 부족."
    else:
        related_section = "\n\n**관련 규정**:\n정보 부족. 추가 문서를 제시해 주세요."

    # ✅ 최종 결합
    answer = body_answer.strip() + related_section
    if "정보 부족" in answer or len(sel) == 0:
        answer += "\n\n> 추가로 필요한 키워드나 문서 범위를 지정해 주세요."

    return {
        "messages": state["messages"] + [AIMessage(content=answer)],
        "candidate_answer": answer,
        "retries": state.get("retries", 0) + 1,
    }
