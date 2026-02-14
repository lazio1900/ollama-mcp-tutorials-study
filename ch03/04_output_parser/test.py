from openai import OpenAI

# base_url 끝에 /v1 이 반드시 포함되어야 합니다.
client = OpenAI(
    base_url="https://kl2h3oy6yd1v7u-8000.proxy.runpod.net/v1", 
    api_key="sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86"
)
# 2. 모델명 (출력된 이름 그대로)
model_name = "openai/gpt-oss-20b"

print(f"[{model_name}] 모델에 텍스트 완성을 요청합니다...")

# 3. Chat이 아닌 Completions 사용
try:
    response = client.completions.create(
        model=model_name,
        prompt="인공지능의 미래는",  # 질문이 아니라 문장의 시작을 줍니다.
        max_tokens=100,
        temperature=0.7
    )

    # 4. 결과 출력
    print("\n--- 결과 ---")
    print(response.choices[0].text)

except Exception as e:
    print(f"에러 발생: {e}")