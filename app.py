from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
import re

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
title="Text Summarizer App",
description="Lightweight Text Summarization App",
version="1.0"
)

summarizer = pipeline(
"summarization",
model="sshleifer/distilbart-cnn-12-6",
device=-1
)

templates = Jinja2Templates(directory="templates")

app.mount(
"/static",
StaticFiles(directory="static"),
name="static"
)

class DialogueInput(BaseModel):
dialogue: str

def clean_data(text: str):

text = re.sub(r"\r\n", " ", text)
text = re.sub(r"\s+", " ", text)
text = re.sub(r"<.*?>", " ", text)

return text.strip()

def summarize_dialogue(dialogue: str) -> str:

dialogue = clean_data(dialogue)

dialogue = dialogue[:1000]

summary = summarizer(
    dialogue,
    max_length=60,
    min_length=15,
    do_sample=False
)

return summary[0]["summary_text"]

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

summary = summarize_dialogue(
    dialogue_input.dialogue
)

return {
    "summary": summary
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={
        "request": request
    }
)