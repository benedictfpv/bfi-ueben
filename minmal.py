# run_chatbot.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

local_dir = "./phi3-4bit"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    device_map={"": device},
)

chat = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    do_sample=False,
)

system_prompt = "You are a helpful assistant. Reply concisely to the user in 1–2 sentences. "
history = []

def format_messages(system, history, user_msg):
    msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_msg}]
    text = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in msgs)
    text += "<|assistant|>\n"
    return text

while True:
    user = input("You: ")
    if user.strip().lower() in {"quit", "exit"}:
        break
    prompt = format_messages(system_prompt, history, user)
    out = chat(prompt)[0]["generated_text"]
    reply = out.split("<|assistant|>")[-1].strip()
    reply = reply.split("####")[0]
    print(f"Bot: {reply}\n")
    history.append({"role": "user", "content": user})
    history.append({"role": "assistant", "content": reply})
