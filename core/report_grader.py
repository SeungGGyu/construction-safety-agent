import sys
import logging
from dotenv import load_dotenv
import os
import requests

# 상위 경로 import 가능하도록
sys.path.append('..')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv('.env')

# API 설정 변수들
MODEL_TOKEN = os.getenv("MODEL_TOKEN")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URI") 
MODEL_NAME = os.getenv("MODEL_NAME")



def grade_report_quality(state: dict) -> str:
    report = state.get("report", "")
    question = """
    다음 건설안전 보고서가 충분히 완전한가?
    주요 항목(목적, 원인, 법규, 대책)이 모두 다뤄졌는지 평가하라.
    부족하면 'insufficient', 충분하면 'adequate'로만 답하라.
    """
    response = requests.post(f"{LLM_URL}/v1/chat/completions", json={
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": question}, {"role": "user", "content": report}]
    }).json()
    verdict = response["choices"][0]["message"]["content"].lower()
    return "insufficient" if "insufficient" in verdict else "adequate"
