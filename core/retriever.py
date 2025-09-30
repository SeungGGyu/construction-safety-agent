# retriever.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.schema import Document
from typing import Dict, Any, List


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
        ensemble_weights: tuple = (0.7, 0.3),
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
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        content_db = FAISS.load_local(
            self.faiss_db_path, embeddings, allow_dangerous_deserialization=True
        )

        dense_retriever = content_db.as_retriever(
            search_type="similarity", search_kwargs={"k": self.top_k}
        )

        all_docs = list(content_db.docstore._dict.values())
        sparse_retriever = BM25Retriever.from_documents(all_docs)
        sparse_retriever.k = self.top_k

        hybrid_retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            weights=list(self.ensemble_weights),
        )

        cross_encoder = HuggingFaceCrossEncoder(model_name=self.reranker_model)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=self.top_k)

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
    embedding_model="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    reranker_model="BAAI/bge-reranker-v2-m3",
    top_k=8,
    ensemble_weights=(0.2, 0.8),
)

def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    docs = retriever_instance.retrieve(query)

    # 컨텍스트 문자열 합치기 (인용번호 붙이기)
    docs_text = "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
    )

    return {
        "retrieved": docs,
        "selected": docs,
        "docs_text": docs_text,
        "sources": [
            {"idx": i+1, "title": doc.metadata.get("title", ""), "url": doc.metadata.get("url", "")}
            for i, doc in enumerate(docs)
        ],
    }

