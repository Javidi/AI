import transformers
model_id = "Qwen/Qwen2.5-3B-Instruct"

pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    load_in_4bit=True,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

while True:
    prompt = input("prompt: ")

    if prompt == "exit":
        break
    messages.append(prompt)
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )

    print(outputs)
    
    messages.append(outputs)
