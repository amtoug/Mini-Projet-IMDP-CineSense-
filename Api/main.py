from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = None
classifier = None


@app.on_event("startup")
def load_models():
    global analyzer, classifier

    # VADER
    analyzer = joblib.load("../model/vader_analyzer.pkl")

    # DistilBERT
    classifier = pipeline(
        "sentiment-analysis",
        model="../model/distilbert_model",
        tokenizer="../model/distilbert_model",
        device=-1
    )


class InputData(BaseModel):
    text: str


@app.post("/predict")
def predict(data: InputData):
    scores = analyzer.polarity_scores(data.text)

    if scores["compound"] >= 0.1:
        sentiment = 1
    elif scores["compound"] <= -0.1:
        sentiment = -1
    else:
        sentiment = 0

    return {
        "sentiment": sentiment,
        "scores": scores
    }


@app.post("/predict_bert")
def predict_bert(data: InputData):
    result = classifier(data.text)[0]

    return {
        "label": result["label"],
        "score": result["score"]
    }
