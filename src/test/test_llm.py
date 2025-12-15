from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama


def main():
    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    messages = [
        SystemMessage(content="너는 모든 질문에 반드시 한국어로 답변해야 해."),
        HumanMessage(content="RAG (Retrieval-Argmented Generation) 가 뭔지 한 문장으로 설명해줘.")
    ]

    responses = llm.invoke(messages)
    print(responses.content)

if __name__ == "__main__":
    main()