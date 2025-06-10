from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Path to your local model folder
model_path = "/modeller/Nosus-Hermes-2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

# Set up the pipeline
chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Simple REPL loop
print("Chat with Nous Hermes 2 (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat(user_input, max_new_tokens=200, do_sample=True, temperature=0.7)[0]['generated_text']
    # Only show model's reply
    print("AI:", response[len(user_input):].strip())

