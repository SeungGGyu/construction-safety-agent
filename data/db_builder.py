

import os
import json
import pickle
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# ===== 1️⃣ 경로 설정 =====
base_dir = "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent"
chunk_dir = os.path.join(base_dir, "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/chunks")  # ✅ chunk 파일 저장 폴더
save_dir = os.path.join(base_dir, "/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB")  # ✅ 저장 경로

# ===== 2️⃣ Qwen Embedding API 설정 =====
embedder_model_name = "Qwen/Qwen3-Embedding-4B"
embedder_base_url = "http://211.47.56.71:15653/v1"
embedder_api_key = "token-abc123"       

embedding_model = OpenAIEmbeddings(
    model=embedder_model_name,
    base_url=embedder_base_url,
    api_key=embedder_api_key
)

# ===== 3️⃣ chunk 파일 불러오기 =====
chunk_json_path = os.path.join(chunk_dir, "chunks.json")
chunk_pkl_path = os.path.join(chunk_dir, "chunks.pkl")

chunks = []

if os.path.exists(chunk_json_path):
    print(f"📂 JSON 파일에서 청크 불러오는 중: {chunk_json_path}")
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

elif os.path.exists(chunk_pkl_path):
    print(f"📂 PKL 파일에서 청크 불러오는 중: {chunk_pkl_path}")
    with open(chunk_pkl_path, "rb") as f:
        chunks = pickle.load(f)
else:
    raise FileNotFoundError("❌ chunk.json 또는 chunk.pkl 파일이 존재하지 않습니다.")

print(f"✅ 총 로드된 청크 수: {len(chunks)}")

# ===== 4️⃣ LangChain 문서 형식으로 변환 =====
docs = []
for c in tqdm(chunks, desc="🧩 Document 변환 중"):
    # chunk 구조 예시:
    # {"content": "...", "source": "file.md", "section": "제1장 총칙"}
    text = c.get("content") or c.get("page_content") or ""
    meta = {
        "source": c.get("source", "unknown"),
        "section": c.get("section", c.get("heading", "본문"))
    }
    docs.append(Document(page_content=text, metadata=meta))

print(f"✅ LangChain 문서 개체로 변환 완료: {len(docs)}개")

# ===== 5️⃣ FAISS 벡터 DB 생성 =====
print("🧠 임베딩 및 벡터 생성 중... (Qwen3-Embedding-4B)")
db = FAISS.from_documents(docs, embedding_model)

# ===== 6️⃣ DB 저장 =====
os.makedirs(save_dir, exist_ok=True)
db.save_local(save_dir)

print(f"\n✅ RAG 벡터 DB 구축 완료!")
print(f"📁 저장 위치: {save_dir}")
print(f"🧩 총 청크 수: {len(docs)}")

# ===== 7️⃣ 테스트 검색 =====
query = "이동식작업대차 안전조치"
results = db.similarity_search(query, k=3)

print("\n🔍 테스트 검색 결과:")
for r in results:
    print(f"\n[파일] {r.metadata.get('source', '')} | [섹션] {r.metadata.get('section', '')}")
    print(r.page_content[:300], "...\n")
