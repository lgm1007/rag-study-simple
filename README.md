# 🤖 RAG MVP 실습
### 프로젝트 개요
- 로컬 환경에서 RAG 최소 기능 실습해보기 위한 프로젝트
- 문서 → 검색 → LLM에게 질문하여 답변 받아보기

### 기술 스택
- Python 3.10.14
- LangChain
- 벡터 DB: FAISS
- 임베딩 모델: sentence-transformers
- LLM: Ollama + Llama3

### 실습 준비
Python 설치 후

```
pip install \
  langchain \
  langchain-community \
  langchain-ollama \
  ollama \
  faiss-cpu \
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
