# core/llm_utils.py
import requests
from typing import List, Dict, Any

# === Qwen API 설정 (공통) ===
LLM_BASE = "http://211.47.56.71:8908/v1" 
LLM_MODEL = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
LLM_TOKEN = "token-abc123"


def call_llm(messages: List[Dict[str, str]],
             temperature: float = 0.3,
             top_p: float = 0.9,
             max_tokens: int = 2000) -> str:
    """
    모든 노드에서 사용할 공통 LLM 호출 유틸
    (LangChain 불필요 — 로컬 Qwen API 직접 호출)
    """
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    headers = {
        "Authorization": f"Bearer {LLM_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        # ✅ 반드시 /v1/chat/completions 로 요청
        response = requests.post(
            f"{LLM_BASE}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=180
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ LLM 호출 실패: {e}")
        if "response" in locals():
            print(f"서버 응답: {response.text}")
        return "⚠️ 모델 응답 실패. 다시 시도해주세요."


def simple_chat(prompt: str, system: str = "당신은 건설 안전 지침 전용 어시스턴트입니다.") -> str:
    """단일 프롬프트용 간단 호출 버전"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return call_llm(messages)
