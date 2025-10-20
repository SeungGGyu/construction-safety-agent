from transformers import AutoModel, AutoTokenizer
import torch
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from typing import Dict, Any, List


# === Qwen 전용 Embeddings 클래스 ===
class QwenEmbeddings(Embeddings):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B", output_dim=768):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.output_dim = output_dim

    def _mean_pool(self, outputs, inputs):
        # CLS/SEP 무시하고 평균 pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled = self._mean_pool(outputs, inputs)
            pooled = pooled[:, :self.output_dim]  # ✅ DB 차원에 맞게 슬라이싱
            vectors = pooled.to(torch.float32).cpu().numpy()
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        inputs = self.tok([text], padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled = self._mean_pool(outputs, inputs)
            pooled = pooled[:, :self.output_dim]
            vector = pooled.to(torch.float32).cpu().numpy()[0]
        return vector.tolist()


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
    embedding_model="Qwen/Qwen3-Embedding-4B",
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
