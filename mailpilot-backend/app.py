from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI(title="MailPilot AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
LORA_PATH = "../MailPilot-AI"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)

model = model.to("cpu")
model.eval()

print("MailPilot AI is ready!")


class GenerateRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    return {"message": "MailPilot AI Backend is running 🚀"}


@app.post("/generate")
def generate(request: GenerateRequest):

    inputs = tokenizer(
        request.prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {
        "response": response
    }
