# 🤖 RAG MVP 실습
## 프로젝트 개요
- 로컬 환경에서 RAG 최소 기능 실습해보기 위한 프로젝트
- 문서 → 검색 → LLM에게 질문하여 답변 받아보기

## 기술 스택
- Python 3.10.14
- LangChain
- 벡터 검색 엔진 (벡터 검색 실습용): FAISS
- 벡터스토어: Qdrant
- 임베딩 모델: Hugging Face [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- LLM: Ollama + Llama3

## 실습 준비
Python 설치 후

1) 가상환경 생성 및 활성화
```
python -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

2) 필요 패키지 설치
```
pip install \
  langchain \
  langchain-community \
  langchain-ollama \
  langchain-qdrant \
  ollama \
  faiss-cpu \
  qdrant-client \
  sentence-transformers \
  python-dotenv
```

https://ollama.com 사이트에서 OS별 앱 설치

| 모델           | 이유              |
| ------------ | --------------- |
| `llama3:8b`  | 기본 성능 좋고 안정적    |
| `qwen2.5:7b` | 한글 질문에 상대적으로 강함 |
| `phi-3`      | 매우 가볍고 빠름       |

원하는 모델 설치

```
> ollama run llama3
```

모델 설치 끝나면 설치 확인

![](https://i.imgur.com/bOmIvu0.png)

3) Qdrant 로컬 Docker 설치 및 실행

도커 설치 후 Qdrant 이미지 Pull & Run

```
mkdir -p qdrant_storage

docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

## 실행
### 1) [ingest.py](src/main/ingest.py)
data/raw 경로에 들어있는 문서 내용으로 벡터 DB 생성

```
python src/main/ingest.py
```

프로젝트 루트/vector_db 경로 하위에 벡터 DB 파일이 생성되면 성공

![](https://i.imgur.com/aC2ePf5.png)

### 2) [qa.py](src/main/qa.py)
생성된 벡터 DB에서 사용자에게 받은 질문 내용을 검색하여 해당 결과를 기반으로 답변 생성

```
python src/main/qa.py
```

실행 예시

![](https://i.imgur.com/NUSeM4J.png)
