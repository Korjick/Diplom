import uvicorn
from fastapi import FastAPI, HTTPException, Request
import numpy as np
from pydantic import BaseModel
import torch

from NeuralNetwork import NeuralNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Executed on:', device)

nn_model_loaded = NeuralNetwork().to(device)
nn_model_loaded.load_state_dict(torch.load("nn_model_2.pth"))

app = FastAPI()


class InputData(BaseModel):
    data: list


@app.post("/predict")
async def predict(input_data: InputData):
    data = np.array(input_data.data, dtype=np.float32)

    if data.shape[0] != 10:
        raise HTTPException(status_code=400, detail="Invalid input shape, expected 10 features")

    print(type(data))
    X_test_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    last_result = nn_model_loaded(X_test_tensor).detach().cpu().numpy().tolist()
    return {"predictions": last_result}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)