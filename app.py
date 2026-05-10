from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
import re
import torch

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ---------------- FASTAPI APP ----------------

app = FastAPI(
title="Text Summarizer App",
description="Lightweight Text Summarization App",
version="1.0"
)

# ---------------- DEVICE ----------------

device = -1   # CPU ONLY

# ---------------- LOAD LIGHTWEIGHT MODEL ----------------

summarizer = pipeline(
"summarization",
model="sshleifer/distilbart-cnn-12-6",
device=device
)

# ---------------- TEMPLATES ----------------

templates = Jinja2Templates(
directory="templates"
)

# ---------------- STATIC FILES ----------------

app.mount(
"/static",
StaticFiles(directory="static"),
name="static"
)

# ---------------- INPUT SCHEMA ----------------

class DialogueInput(BaseModel):
dialogue: str

# ---------------- CLEAN FUNCTION ----------------

def clean_data(text: str):

```
text = re.sub(r"\r\n", " ", text)
text = re.sub(r"\s+", " ", text)
text = re.sub(r"<.*?>", " ", text)

return text.strip()
```

# ---------------- SUMMARIZATION ----------------

def summarize_dialogue(dialogue: str) -> str:

```
dialogue = clean_data(dialogue)

# limit input size for memory safety
dialogue = dialogue[:1000]

summary = summarizer(
    dialogue,
    max_length=60,
    min_length=15,
    do_sample=False
)

return summary[0]["summary_text"]
```

# ---------------- API ROUTE ----------------

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

```
summary = summarize_dialogue(
    dialogue_input.dialogue
)

return {
    "summary": summary
}
```

# ---------------- HOME PAGE ----------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

```
return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={
        "request": request
    }
)
```
