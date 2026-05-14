# Qwen3 series official API usage example
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-8b",
    messages=[
        {"role": "user", "content": "Who are you?"},
    ],
    extra_body={"enable_thinking": False}
)

# Print model response directly
print(completion.choices[0].message.content)


# ChatGPT / Gemini third-party API usage example
from openai import OpenAI

# Initialize client
client = OpenAI(
    api_key=os.environ.get("LAOZHANG_API_KEY", ""),  # Obtain from laozhang API
    base_url="https://api.laozhang.ai/v1"  # Direct connection address
)

# Example calls for different models
def test_models():
    models = [
        "gpt-5.2",  # ChatGPT
        "gemini-3-flash-preview",  # Gemini
    ]

    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a friendly AI assistant."},
                    {"role": "user", "content": "Introduce yourself in one sentence."}
                ],
                temperature=0.7,
                max_tokens=100
            )
            print(f"{model}: {response.choices[0].message.content}")
            print(f"Tokens used: {response.usage.total_tokens}\n")
        except Exception as e:
            print(f"{model} call failed: {e}\n")

if __name__ == "__main__":
    test_models()
