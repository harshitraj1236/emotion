from contextlib import asynccontextmanager
import pickle
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
1. Constants
"""
# A. Model Path
model_path = "Artifacts/best_model.keras"

# B. Tokenizer Path
tokenizer_path = "Artifacts/tokenizer.pkl"

# C. Config Path (saved in the notebook — max_len + label_names live here,
#    NOT hardcoded, so training and serving can never drift apart silently)
config_path = "Artifacts/config.pkl"

# D. Fallback config, only used if config.pkl is missing. Keep this in sync
#    manually if you ever retrain without regenerating config.pkl.
_FALLBACK_MAX_SEQUENCE_LENGTH = 50
_FALLBACK_EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# E. Emotion emojis
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


"""
2. Request and Response Schemas
"""
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"}
    )

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilites: dict[str, float]

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool


"""
3. Model Loading and Lifespan Management
"""
dl_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model, tokenizer, and config...")
    dl_model["BiGRU"] = load_model(model_path)
    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)

    if os.path.exists(config_path):
        with open(config_path, "rb") as file:
            config = pickle.load(file)
        dl_model["max_len"] = config["max_len"]
        dl_model["labels"] = config["label_names"]
        print(f"Loaded config.pkl -> max_len={dl_model['max_len']}, labels={dl_model['labels']}")
    else:
        print(
            "WARNING: Artifacts/config.pkl not found — falling back to hardcoded "
            "max_len/labels. Regenerate and deploy config.pkl (notebook cell 17) "
            "to remove this risk."
        )
        dl_model["max_len"] = _FALLBACK_MAX_SEQUENCE_LENGTH
        dl_model["labels"] = _FALLBACK_EMOTION_LABELS

    print("Models loaded successfully...")
    yield
    dl_model.clear()


"""
4. FastAPI App Initialization & Middleware
"""
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


"""
5. API Endpoints
"""
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse("static/index.html")

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))

@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later."
        )

    max_sequence_length = dl_model["max_len"]
    emotion_labels = dl_model["labels"]

    # IMPORTANT: no manual regex cleaning here. During training, the raw text
    # was fed straight into tokenizer.texts_to_sequences() (see notebook
    # cells 16 & 18) — the Tokenizer's own saved filters/lower settings did
    # all the cleaning. In particular, the default Tokenizer filters do NOT
    # strip apostrophes, so "can't" was tokenized as "can't", not "cant".
    # Re-cleaning text here with a different rule set (as the old code did)
    # produces tokens the tokenizer never saw during training and pushes
    # them to <OOV>. Just strip incidental whitespace and let the tokenizer
    # apply the exact same transformation it learned from.
    cleaned_text = text_input.text.strip()

    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    probabilities = BiGRU_model.predict(padded_sequence)[0]
    top_emotion_index = int(np.argmax(probabilities))

    all_probabilities = {
        label: float(prob)
        for label, prob in zip(emotion_labels, probabilities)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilites=all_probabilities
    )