from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

KANANA_ID = "kakaocorp/kanana-1.5-8b-instruct-2505"

def make_kanana_llm(
    model_id: str = KANANA_ID,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # bf16 지원 안 하는 경우를 대비해 자동 폴백
    try:
        dtype = torch.bfloat16
        _ = torch.zeros(1, device="cuda", dtype=dtype)  # 테스트
    except Exception:
        dtype = torch.float16

    # GPU 0번으로 직접 로드 (단일 GPU 사용)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda:1")

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        device=1,                     # 파이프라인도 GPU 0 사용
        return_full_text=False,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
    )

    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen))

# ✅ Kanana ChatModel 객체 생성
KANANA = make_kanana_llm()


