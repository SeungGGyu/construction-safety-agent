from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
import torch

# === LangChain Core ===
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# === VectorStore ===
from langchain_community.vectorstores import FAISS

# === Retrievers ===
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# === Reranker / Compressor ===
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from typing import Dict, Any, List
import numpy as np


# === 🔹 외부 Qwen3 임베딩 API 설정 ===
embedder_model_name = "Qwen/Qwen3-Embedding-4B"
embedder_base_url = "http://211.47.56.71:15653/v1"
embedder_api_key = "token-abc123"

embed_client = OpenAI(
    base_url=embedder_base_url,
    api_key=embedder_api_key
)


# === Qwen 임베딩 API 기반 Embeddings 클래스 ===
class QwenEmbeddings(Embeddings):
    """
    Qwen3 임베딩 API 호출 기반 Embeddings 클래스
    (로컬 모델 로딩 대신 HTTP 요청으로 벡터 생성)
    """

    def __init__(self, model_name=embedder_model_name, output_dim=768):
        self.model_name = model_name
        self.output_dim = output_dim

    def _get_embedding(self, text: str) -> List[float]:
        """단일 텍스트에 대한 임베딩 요청"""
        response = embed_client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding[: self.output_dim]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서에 대한 임베딩 요청"""
        embeddings = []
        for text in texts:
            vec = self._get_embedding(text)
            embeddings.append(vec)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """질문(query)에 대한 임베딩 요청"""
        return self._get_embedding(text)


# === RerankRetriever 정의 ===
class RerankRetriever:
    """
    Hybrid Retriever(Dense + BM25 + Cross-Encoder Reranker)
    """

    def __init__(
        self,
        faiss_db_path: str,
        embedding_model: str,
        reranker_model: str,
        top_k: int = 10,
        ensemble_weights: tuple = (0.5, 0.5),
    ):
        self.faiss_db_path = faiss_db_path
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.top_k = top_k
        self.ensemble_weights = ensemble_weights
        self.retriever = None

        print(f"🔍 RerankRetriever 초기화 중 (top_k={self.top_k})")
        self._setup()
        print("✅ RerankRetriever 생성 완료")

    def _setup(self):
        embeddings = QwenEmbeddings(model_name=self.embedding_model)

        # === Dense (FAISS)
        content_db = FAISS.load_local(
            self.faiss_db_path, embeddings, allow_dangerous_deserialization=True
        )
        dense_retriever = content_db.as_retriever(
            search_type="similarity", search_kwargs={"k": self.top_k}
        )

        # === Sparse (BM25)
        all_docs = list(content_db.docstore._dict.values())
        sparse_retriever = BM25Retriever.from_documents(all_docs)
        sparse_retriever.k = self.top_k

        # === Ensemble
        hybrid_retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            weights=list(self.ensemble_weights),
        )

        # === Cross-Encoder Reranker
        cross_encoder = HuggingFaceCrossEncoder(model_name=self.reranker_model)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=self.top_k)

        # === 최종 ContextualCompressionRetriever
        self.retriever = ContextualCompressionRetriever(
            base_retriever=hybrid_retriever,
            base_compressor=compressor,
        )

    def retrieve(self, query: str) -> List[Document]:
        print(f"\n📝 입력 쿼리: {query}")
        return self.retriever.get_relevant_documents(query)


# === LangGraph용 Node 함수 ===
retriever_instance = RerankRetriever(
    faiss_db_path="/home/user/Desktop/jiseok/capstone/RAG/DB/construction_safety_guidelines_faiss",
    embedding_model=embedder_model_name,
    reranker_model="BAAI/bge-reranker-v2-m3",
    top_k=8,
    ensemble_weights=(0.3, 0.7),
)


def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    docs = retriever_instance.retrieve(query)

    # 본문 + 파일명/페이지 같이 표시
    docs_text = "\n\n".join(
        f"[{i+1}] ({doc.metadata.get('filename','?')} p.{doc.metadata.get('page','?')})\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    # sources: filename/page/idx만 간결하게 정리
    sources = [
        {
            "idx": i + 1,
            "filename": doc.metadata.get("filename", ""),
            "page": doc.metadata.get("page", ""),
        }
        for i, doc in enumerate(docs)
    ]

    return {
        "retrieved": docs,
        "selected": docs,
        "docs_text": docs_text,
        "sources": sources,
    }
