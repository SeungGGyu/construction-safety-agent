# core/llm_utils.py
import os
import requests
from typing import List, Dict, Any

# === ✅ 환경 변수 기반 설정 (없으면 기본값으로 대체) ===
API_KEY = os.environ.get("MODEL_TOKEN", "token-abc123")
BASE_URL = os.environ.get("MODEL_BASE_URL", "http://211.47.56.71:8908/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B-GPTQ-Int4")

# === Qwen API 설정 (공통) ===
LLM_BASE = BASE_URL.rstrip("/")  # 혹시 슬래시 중복 방지
LLM_MODEL = MODEL_NAME
LLM_TOKEN = API_KEY


def call_llm(messages: List[Dict[str, str]],
             temperature: float = 0.3,
             top_p: float = 0.9,
             max_tokens: int = 2000) -> str:
    """
    모든 노드에서 사용할 공통 LLM 호출 유틸
    (LangChain 불필요 — 로컬 Qwen API 직접 호출)

    Args:
        messages: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        temperature: 생성 다양성 조절
        top_p: nucleus sampling
        max_tokens: 최대 토큰 수

    Returns:
        LLM이 생성한 문자열 (content)
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
        # ✅ 반드시 OpenAI 호환 엔드포인트로 요청
        response = requests.post(
            f"{LLM_BASE}/chat/completions"
            if not LLM_BASE.endswith("/v1") else
            f"{LLM_BASE}/chat/completions",
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
    """
    단일 프롬프트용 간단 호출 버전
    (예: rewrite, grader 등에서 간단히 사용 가능)
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return call_llm(messages)
