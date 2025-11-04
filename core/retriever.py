import os
from typing import Dict, Any, List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker


# === Qwen API 기반 Embedding 클래스 ===
def get_qwen_api_embeddings():
    """
    Qwen3-Embedding-4B API 호출 기반 Embedding
    """
    embedder_model_name = "Qwen/Qwen3-Embedding-4B"
    embedder_base_url = "http://211.47.56.71:15653/v1"
    embedder_api_key = "token-abc123"

    print(f"🌐 Qwen Embedding API 연결 중: {embedder_base_url}")
    embeddings = OpenAIEmbeddings(
        model=embedder_model_name,
        base_url=embedder_base_url,
        api_key=embedder_api_key,
    )
    return embeddings


# === RerankRetriever 정의 ===
class RerankRetriever:
    """
    Hybrid Retriever(Dense + BM25 + Cross-Encoder Reranker)
    """

    def __init__(
        self,
        faiss_db_path: str,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        top_k: int = 10,
        ensemble_weights: tuple = (0.5, 0.5),
    ):
        self.faiss_db_path = faiss_db_path
        self.reranker_model = reranker_model
        self.top_k = top_k
        self.ensemble_weights = ensemble_weights
        self.retriever = None

        print(f"🔍 RerankRetriever 초기화 중 (top_k={self.top_k})")
        self._setup()
        print("✅ RerankRetriever 생성 완료")

    def _setup(self):
        # === 1️⃣ Qwen API Embeddings ===
        embeddings = get_qwen_api_embeddings()

        # === 2️⃣ FAISS DB 로드 ===
        if not os.path.exists(self.faiss_db_path):
            raise FileNotFoundError(f"❌ DB 경로를 찾을 수 없습니다: {self.faiss_db_path}")

        content_db = FAISS.load_local(
            self.faiss_db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # === 3️⃣ Dense Retriever (FAISS) ===
        dense_retriever = content_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        # === 4️⃣ Sparse Retriever (BM25) ===
        all_docs = list(content_db.docstore._dict.values())
        sparse_retriever = BM25Retriever.from_documents(all_docs)
        sparse_retriever.k = self.top_k

        # === 5️⃣ Hybrid Retriever (Dense + Sparse) ===
        hybrid_retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            weights=list(self.ensemble_weights),
        )

        # === 6️⃣ Cross-Encoder Reranker ===
        cross_encoder = HuggingFaceCrossEncoder(model_name=self.reranker_model)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=self.top_k)

        # === 7️⃣ Contextual Compression Retriever ===
        self.retriever = ContextualCompressionRetriever(
            base_retriever=hybrid_retriever,
            base_compressor=compressor,
        )

    def retrieve(self, query: str) -> List[Document]:
        print(f"\n📝 입력 쿼리: {query}")
        return self.retriever.get_relevant_documents(query)


# === LangGraph용 Node 함수 ===
retriever_instance = RerankRetriever(
    faiss_db_path="/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/DB",
    reranker_model="BAAI/bge-reranker-v2-m3",
    top_k=8,
    ensemble_weights=(0.5, 0.5),
)


def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    docs = retriever_instance.retrieve(query)

    # 본문 + 파일명/페이지 같이 표시
    docs_text = "\n\n".join(
        f"[{i+1}] ({doc.metadata.get('filename','?')} p.{doc.metadata.get('page','?')})\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    # sources 정리
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
