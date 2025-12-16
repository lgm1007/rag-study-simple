import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# src/main/ingest.py 기준: 프로젝트 루트는 3단계 위
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")

def load_documents(data_dir: str) -> List:
    docs = []

    for root, _, files in os.walk(data_dir):
        for fname in files:
            path = os.path.join(root, fname)
            # 확장자 추출
            ext = os.path.splitext(fname)[1].lower()

            if ext == ".pdf":
                # PyPDFLoader는 페이지 단위로 Document를 생성함
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif ext == ".txt":
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
            else:
                # 필요하면 다른 문서도 확장 가능
                continue
    return docs

def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    # 1. 문서 로딩 (data/raw 하위 문서 전부)
    docs = load_documents(DATA_DIR)
    if not docs:
        raise FileNotFoundError(f"No documents found in {DATA_DIR}")

    # 2. 청크 분할
    # PDF는 페이지 단위로 쪼개져 들어오는 경우가 많아서 너무 작은 chunk로 쪼개지지 않도록 기본값을 약간 크게 잡아도 됨
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
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
