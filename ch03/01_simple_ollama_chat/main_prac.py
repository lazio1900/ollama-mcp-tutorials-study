from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

POD_ID = "rauik3dn1og4en"
BASE = f"https://{POD_ID}-11434.proxy.runpod.net"

llm = ChatOllama(model="qwen3:8b", base_url = BASE)

while True:
    user_input = input("질문을 입력하세요 (종료: exit): ")
    if user_input.lower() == "exit":
        break

    messages = [HumanMessage(content=user_input)]

    response = llm.invoke(messages)

    print("답변:", response.content)