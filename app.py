from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# ---------------- FASTAPI APP ---------------- #

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)


# ---------------- LOAD MODEL ---------------- #

model_path = "./saved_summary_model"

MODEL_NAME = "t5-small"

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


# ---------------- DEVICE SETUP ---------------- #

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

model.to(device)


# ---------------- TEMPLATES ---------------- #

templates = Jinja2Templates(
    directory="templates"
)


# ---------------- STATIC FILES ---------------- #

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ---------------- INPUT SCHEMA ---------------- #

class DialogueInput(BaseModel):
    dialogue: str


# ---------------- CLEAN FUNCTION ---------------- #

def clean_data(text: str):

    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    return text.strip().lower()


# ---------------- SUMMARIZATION ---------------- #

def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    # T5 task prefix
    dialogue = "summarize: " + dialogue

    # Tokenization
    inputs = tokenizer(
        dialogue,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move tensors to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate summary
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    # Decode summary
    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# ---------------- API ROUTE ---------------- #

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }


# ---------------- HOME PAGE ---------------- #

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )