"""
Hello Local AI (Ollama)
---------------------------------
Goal: prove your setup works. Send ONE question to a local AI model and print
the answer. Runs 100% on your machine — no API key, no cost, no internet.

Before running:
  1. Install Ollama from https://ollama.com/download
  2. In a terminal, download a model:  ollama pull llama3.2
  3. Then run this:  python examples/hello_ollama.py
"""

import ollama

MODEL = "llama3.2"


def ask_ai(question: str) -> str:
    """Send a single question to the local model and return the answer."""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return response["message"]["content"]


def main():
    print(f"🤖 Ask the local AI anything (model: {MODEL}). Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Bye! 👋")
            break
        if not question:
            continue

        try:
            answer = ask_ai(question)
        except Exception as e:
            print(
                f"\n❌ Couldn't reach the model: {e}\n"
                "Is Ollama installed and running? Did you run `ollama pull llama3.2`?\n"
            )
            continue

        print(f"\nAI: {answer}\n")


if __name__ == "__main__":
    main()
