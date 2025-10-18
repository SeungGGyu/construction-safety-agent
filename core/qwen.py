# qwen.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM   # GPTQ 전용 로더

QWEN_ID = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"

def make_qwen_llm(model_id=QWEN_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        device="cuda:0",
        trust_remote_code=True,
        use_safetensors=True,
     )
    # pipeline 생성 (langchain-huggingface와 호환)
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    return HuggingFacePipeline(pipeline=pipe)


# ? Qwen ChatModel 객체
QWEN = make_qwen_llm()
