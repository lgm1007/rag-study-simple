import csv
import os
import re
from typing import List, Dict, Tuple, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES_CSV = os.path.join(PROJECT_ROOT, "data", "files.csv")

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "company_docs" # ingest.py에서 정의한 컬렉션 이름

SYSTEM_PROMPT = """
    너는 RAG 기반 한국어 어시스턴트다.
    
    규칙:
    1) 반드시 한국어로만 답한다.
    2) 주어진 [컨텍스트]에 근거해서만 답한다.
    3) 컨텍스트에 답이 없으면 "제공된 문서에는 해당 정보가 없습니다." 라고 답한다.
    4) 추측하거나 지어내지 않는다.
    5) 컨텍스트라는 내용은 답변에서 언급하지 않는다.
"""

def build_prompt(context: str, question: str) -> str:
    return f"""[컨텍스트]
    {context}
    
    [질문]
    {question}
    
    [답변]
"""

# csv 파일에서 회사 목록 로드
def load_company(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"files.csv not found in {csv_path}")

    companies: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company_name = (row.get("company_name") or "").strip()
            filename = (row.get("filename") or "").strip()
            if company_name:
                companies.append({"company_name": company_name, "filename": filename})
    return companies

def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s) # 공백 제거
    return s

def extract_company_rule_based(question: str, company_names: List[str]) -> Tuple[Optional[str], List[str]]:
    """
    1차: 문자열 규칙 기반 매칭
    - (best) 질문에 회사명이 포함되면 그 회사 선택
    - 여러 개 매칭되면 X
    """
    qn = normalize_text(question)
    matches = []
    for company_name in company_names:
        if normalize_text(company_name) in qn:
            # 질문에 회사명이 포함되면 매칭
            matches.append(company_name)
    if len(matches) == 1:
        # 1개의 회사가 질문에 포함
        return matches[0], matches
    if len(matches) > 1:
        # 2개 이상의 회사가 질문에 포함
        return None, matches
    # 매칭 안 됨
    return None, []

def extract_company_with_llm(question: str, company_names: List[str], llm: ChatOllama) -> Optional[str]:
    """
    2차: LLM에게 질문을 바탕으로 목록 중 한가지 선택하도록 함
    - 없으면 None
    - 출력은 company_name 한 줄만
    """
    candidates = "\n".join([f"- {c}" for c in company_names])
    router_system = """
        너는 LLM 시스템의 라우터이다.
        규칙:
        - 사용자 질문에서 어떤 '회사'를 대상으로 한 질문인지 판단한다.
        - 반드시 아래 후보 회사 목록 중 하나만 출력하거나, 후보에 없으면 None만 출력한다.
        - 질문에 후보 회사 목록에 존재하는 회사명이 명시적으로 포함되어 있지 않다면 반드시 None을 출력한다.
        - 추측/유추/가정 금지.
        - 출력은 오직 회사명 1줄 또는 None 1줄. 
    """

    router_user = f"""
        [후보 회사 목록]
        {candidates}
        
        [사용자 질문]
        {question}
        
        출력 형식(반드시 지켜야함):
        - 후보 중 하나만: 상품명 그대로 한 줄
        - 없으면: None
    """

    resp = llm.invoke([
        SystemMessage(content=router_system),
        HumanMessage(content=router_user)
    ]).content.strip()

    # 목록에 있는 값만 인정
    if resp in company_names:
        return resp
    return None

def make_company_filter(company_name: str) -> Filter:
    # 회사명 일치 조건 필터
    return Filter(
        must=[
            FieldCondition(key="metadata.company_name", match=MatchValue(value=company_name))
        ]
    )

def main():
    # 1) 회사 목록 로드
    companies = load_company(FILES_CSV)
    company_names = sorted(list({c["company_name"] for c in companies}))

    # 2) 임베딩 모델
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Qdrant 연결
    client = QdrantClient(QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    # 3) LLM (Ollama)
    llm = ChatOllama(model="llama3", temperature=0)

    # 4) 질문 입력 루프
    print("RAG QA 시작. 종료하려면 'exit' 입력\n")
    print("※ 질문에 회사명이 포함되면 자동 필터링됩니다.")

    while True:
        try:
            raw = input("Q> ")
        except EOFError:
            print("\nEOF 감지: 종료합니다.")
            break

        question = str(raw).strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        # 5) 회사명 자동 추출 (1차 규칙 후 매칭 안되면 2차 규칙)
        company_name, matches = extract_company_rule_based(question, company_names)

        if company_name is None and matches:
            print("⚠️ 질문에서 회사명이 여러 개 매칭되어 자동 선택이 불가합니다:")
            for m in matches:
                print(" -", m)

        if company_name is None:
            # 2차 규칙으로 회사명 찾기
            company_name = extract_company_with_llm(question, company_names, llm)

        if company_name is None:
            print("⚠️ 질문에서 회사명을 특정할 수 없습니다.")

        # 6) 검색 시 회사명 메타데이터 필터링
        if company_name is None:
            retrieved = vector_store.max_marginal_relevance_search( # MMR 적용
                question,
                k=5, # 최종 반환 개수
                fetch_k=30, # 후보 풀
                lambda_mult=0.6 # 질문과의 유사도 (1에 가까울수록 질문과 얼마나 비슷한지를 중요시함)
            )
        else:
            question_filter = make_company_filter(company_name)
            retrieved = vector_store.max_marginal_relevance_search(
                question,
                k=5,
                fetch_k=30,
                lambda_mult=0.6,
                filter=question_filter
            )

        context = "\n\n".join([f"- {d.page_content}" for d in retrieved])

        # 7) 답변 생성
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_prompt(context, question))
        ]
        answer = llm.invoke(messages).content

        print("\nA> ", answer)
        print("\n--- (검색된 컨텍스트) ---")
        for i, d in enumerate(retrieved, 1):
            source = d.metadata.get("source", "")
            page = d.metadata.get("page", None)
            page_str = f" (page {page})" if page is not None else ""
            preview = d.page_content.replace("\n", " ")[:160]
            print(f"[{i}] {source}{page_str} | {preview}")
        print("------------------------\n")

if __name__ == "__main__":
    main()
