from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_path = "/modeller/Nous-Hermes-2"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
    device_map="auto",  # accelerate handles it
    torch_dtype="auto"
)

chat = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

print("Chat with Nous Hermes 2 (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    output = chat(user_input, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    print("AI:", output[len(user_input):].strip())

