import csv
import os
from typing import List, Dict, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# src/main/ingest.py 기준: 프로젝트 루트는 3단계 위
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")
FILES_CSV = os.path.join(PROJECT_ROOT, "data", "files.csv")

# files.csv 파일을 로딩하여 파일명 별 메타데이터 가공/추출
def load_file_map(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    filename -> {"company_name": ...}
    """

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"files.csv not found in {csv_path}\n"
            f"data/files.csv 파일을 생성하고 filename,company_name 컬럼을 채워주세요"
        )

    mapping: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"filename", "company_name"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"files.csv 파일의 헤더는 {required} 를 포함해야 합니다. 현재: {reader.fieldnames}")

        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            mapping[filename] = {"company_name": (row.get("company_name") or "").strip()}
    return mapping

def attach_metadata(docs: List, file_path: str, file_map: Dict[str, str]) -> None:
    """
    loader가 만든 Documents들에 metadata를 추가
    주의: splitter로 나누면 이 metadata가 청크들로 그대로 복사됨
    """
    rel_source = os.path.relpath(file_path, PROJECT_ROOT)
    base_name = os.path.basename(file_path)

    for d in docs:
        d.metadata = d.metadata or {}
        # 기본 출처 정보
        d.metadata["source"] = rel_source # data/raw/ACompany.txt
        d.metadata["filename"] = base_name # ACompany.txt

        # 파일 메타데이터
        d.metadata["company_name"] = file_map.get("company_name", "")

def load_documents(data_dir: str, file_map: Dict[str, Dict[str, str]]) -> Tuple[List, List[str]]:
    docs = []
    missing_meta_files: List[str] = []

    for root, _, files in os.walk(data_dir):
        for fname in files:
            path = os.path.join(root, fname)
            # 확장자 추출
            ext = os.path.splitext(fname)[1].lower()

            # 파일별 메타데이터 조회
            file_meta = file_map.get(fname)
            if file_meta is None:
                # 메타데이터 매핑이 없는 파일은 기록해두고 기본값으로 진행
                missing_meta_files.append(fname)
                file_meta = {"company_name": ""}

            if ext == ".pdf":
                # PyPDFLoader는 페이지 단위로 Document를 생성함
                loader = PyPDFLoader(path)
                file_docs = loader.load() # 페이지 단위 Documents
                attach_metadata(file_docs, path, file_meta)
                docs.extend(file_docs)
            elif ext == ".txt":
                loader = TextLoader(path, encoding="utf-8")
                file_docs = loader.load()
                attach_metadata(file_docs, path, file_meta)
                docs.extend(file_docs)
            else:
                # 필요하면 다른 문서도 확장 가능
                continue
    return docs, missing_meta_files

def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    file_map = load_file_map(FILES_CSV)

    # 1. 문서 로딩 (data/raw 하위 문서 전부)
    docs, missing_meta_files = load_documents(DATA_DIR, file_map)
    if not docs:
        raise FileNotFoundError(f"No documents found in {DATA_DIR}")

    if missing_meta_files:
        print("⚠️ files.csv에 메타데이터가 없는 파일들이 있습니다:")
        for f in sorted(set(missing_meta_files)):
            print(" -", f)

    # 2. 청크 분할
    # PDF는 페이지 단위로 쪼개져 들어오는 경우가 많아서 너무 작은 chunk로 쪼개지지 않도록 기본값을 약간 크게 잡아도 됨
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = splitter.split_documents(docs)
    # 디버깅: 첫 청크 메타데이터 확인
    sample = splits[0]
    print("sample metadata =", {
        "filename": sample.metadata.get("filename"),
        "company_name": sample.metadata.get("company_name"),
        "source": sample.metadata.get("source")
    })

    # 3. 로컬 임베딩
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # 4. FAISS 생성 + 저장
    db = FAISS.from_documents(splits, embeddings)
    os.makedirs(DB_DIR, exist_ok=True)
    db.save_local(DB_DIR)

    print(f"Ingest 완료: 문서 {len(docs)} 개, 청크 {len(splits)} 개, 저장: {DB_DIR}")

if __name__ == "__main__":
    main()
