from fastapi import FastAPI
from transformers import BertTokenizer

#import joblib
from utils import load_checkpoint
from BertClassifier import BertClassifier
from pydantic import BaseModel

# Create a FastAPI instance
app = FastAPI()

class EntityPair(BaseModel):
    entity1: str
    entity2: str 

# Load your machine learning model when the app starts
@app.on_event("startup")
async def load_model():
    global model
    model_init = BertClassifier()
    #link to model:https://drive.google.com/drive/folders/1K8A9uf9FoUPwl07TKmo2K0cmN54P__k0?usp=drive_link
    model = load_checkpoint("model/model.pt",model_init)
    

# Define a prediction endpoint
@app.post("/predict")
async def predict(entitypair:EntityPair):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    encoded = tokenizer(entitypair.entity1,entitypair.entity2, padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt")
    mask = encoded['attention_mask'].to("cpu")
    input_id = encoded['input_ids'].squeeze(1).to("cpu")
    output = model(input_id, mask)
    result = output.argmax(dim=1)
    if result==1:
        return("Entities Match")
    else:
        return("Entities Do not Match")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)