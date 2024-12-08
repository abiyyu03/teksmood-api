from typing import Union

from fastapi import FastAPI, Form
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from fastapi.responses import JSONResponse 
from fastapi.encoders import jsonable_encoder
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = [ 
    "http://localhost:5173",
    "http://localhost:8080",
    "https://teksmood.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    teks: str 

@app.post("/emotion/classify")
async def read_model(item: Item):#(text: Annotated[str, Form()]):
    # Tokenisasi input teks 
    model = AutoModelForSequenceClassification.from_pretrained("./emotion-classifier")
    tokenizer = AutoTokenizer.from_pretrained("./emotion-classifier")

    # Tokenisasi input teks
    inputs = tokenizer(item.teks, return_tensors="pt", padding=True, truncation=True)

    # Prediksi
    # with torch.no_grad():  # Tidak perlu menghitung gradien
    #     outputs = model(**inputs)

    # Logits dari model
    # logits = outputs.logits

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = F.softmax(logits, dim=-1)[0]

    # Konversi logits menjadi probabilitas menggunakan softmax
    # probabilities = F.softmax(logits, dim=1)

    # Label emosi
    label_mapping_reverse = {0: "marah", 1: "sedih", 2: "kecewa", 3: "bahagia"}

    # Identifikasi emosi dominan
    # Identifikasi emosi dominan
    max_label_id = torch.argmax(probabilities).item()
    emosi_dominan = label_mapping_reverse[max_label_id]
    persentase_emosi_dominan = round(probabilities[max_label_id].item() * 100, 1)

    # Label dan persentase untuk selain emosi dominan
    emosi_lainnya = {
        label_mapping_reverse[label_id]: round(prob.item() * 100, 2)
        for label_id, prob in enumerate(probabilities)
        if label_id != max_label_id
    }

    print(item.teks)

    # Format JSON yang kompatibel
    json_compatible_item_data = jsonable_encoder({
        "teks": item.teks,
        "emosi_dominan": {
            "label": emosi_dominan,
            "persentase": persentase_emosi_dominan
        },
        "emosi_lainnya": emosi_lainnya
    }) 

    return JSONResponse(content=json_compatible_item_data)
