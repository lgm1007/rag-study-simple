import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")

SYSTEM_PROMPT = """
    너는 RAG 기반 한국어 어시스턴트다.
    
    규칙:
    1) 반드시 한국어로만 답한다.
    2) 주어진 [컨텍스트]에 근거해서만 답한다.
    3) 컨텍스트에 답이 없으면 "제공된 문서에는 해당 정보가 없습니다." 라고 답한다.
    4) 추측하거나 지어내지 않는다.
"""

def build_prompt(context: str, question: str) -> str:
    return f"""[컨텍스트]
    {context}
    
    [질문]
    {question}
    
    [답변]
"""

def main():
    # 1) 임베딩/DB 로드
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

    # 2) LLM (Ollama)
    llm = ChatOllama(model="llama3", temperature=0)

    # 3) 질문 입력 루프
    print("RAG QA 시작. 종료하려면 'exit' 입력\n")
    while True:
        try:
            raw = input("Q> ")
        except EOFError:
            print("\nEOF 감지: 종료합니다.")
            break

        q = str(raw).strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        # 4) 검색
        retrieved = db.max_marginal_relevance_search( # MMR 적용
            q,
            k=5, # 최종 반환 개수
            fetch_k=30, # 후보 풀
            lambda_mult=0.6 # 질문과의 유사도 (1에 가까울수록 질문과 얼마나 비슷한지를 중요시함)
        )
        context = "\n\n".join([f"- {d.page_content}" for d in retrieved])

        # 5) 생성
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_prompt(context, q))
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
