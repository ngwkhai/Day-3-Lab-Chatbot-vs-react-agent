import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.openai_provider import OpenAIProvider


def test_openai_endpoint():
    load_dotenv()
    model = os.getenv("DEFAULT_MODEL", "gpt-4o")
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1 (default)"

    print("--- Testing OpenAI-compatible Provider ---")
    print(f"Endpoint: {base_url}")
    print(f"Model:    {model}")

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set in .env")
        return

    try:
        provider = OpenAIProvider()
        prompt = "Reply with exactly: pong"
        print(f"\nUser: {prompt}")
        result = provider.generate(prompt)
        print(f"Assistant: {result['content']}")
        print(f"Latency: {result['latency_ms']} ms | Tokens: {result['usage']}")
        print("\nProvider is working correctly!")
    except Exception as e:
        print(f"\nError during execution: {e}")


if __name__ == "__main__":
    test_openai_endpoint()
