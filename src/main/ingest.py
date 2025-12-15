import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# src/main/ingest.py 기준: 프로젝트 루트는 3단계 위
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")

def main():
    # 1. 문서 로딩 (data/raw 하위 문서 전부)
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    docs = loader.load()

    # 2. 청크 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    splits = splitter.split_documents(docs)

    # 3. 로컬 임베딩
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. FAISS 생성 + 저장
    db = FAISS.from_documents(splits, embeddings)
    os.makedirs(DB_DIR, exist_ok=True)
    db.save_local(DB_DIR)

    print(f"Ingest 완료: 문서 {len(docs)} 개, 청크 {len(splits)} 개, 저장: {DB_DIR}")

if __name__ == "__main__":
    main()
