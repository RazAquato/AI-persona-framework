from llama_cpp import Llama

# Path to your GGUF model
model_path = "/modeller/Evathene-v1.3/Evathene-v1.3-Q4_K_M.gguf"

# Load the model
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)

# Simple REPL loop
print("Chat with Evathene v1.3 (type 'exit' to quit):")
chat_history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"{chat_history}User: {user_input}\nAssistant:"
    output = llm(prompt, max_tokens=200, stop=["User:", "Assistant:"], echo=False)

    reply = output["choices"][0]["text"].strip()
    print("Evathene:", reply)
    chat_history += f"User: {user_input}\nAssistant: {reply}\n"

