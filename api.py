from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# load model
model = joblib.load("model_alzheimer.pkl")
scaler = joblib.load("scaler_alzheimer.pkl")

class AlzheimerInput(BaseModel):
    age: float  # AGE 
    sex: str  # PTGENDER (Male/Female)
    years_of_education: float
    '''PTEDUCAT
        Completed elementary school - 8 years
        Completed high school - 12 years
        College (undergraduate) - 16 years
        Postgraduate (Master's) - 18 years or more'''
        
    apoe4: int  # APOE4 - genetic risk marker (0, 1 ou 2) 
    MMSE_mini_mental_state_examination: float  # Mini-Mental State Examination (0-100%)
    score_adas13: float  # Alzheimer's Disease Assessment Scale (ADAS13) (0-100%)
    score_cdrsb: float  # Clinical Dementia Rating - Sum of Boxes (0-100%)
    score_first_phase_RAVLT_test: float  # RAVLT Results of the first phase of the auditory memory test (0-100%)
    score_forgetfulness_RAVLT: float  # RAVLT percent forgetting

app = FastAPI()

@app.post("/predict")
def predict(data: AlzheimerInput):
    sex = 1 if data.sex.lower() == "male" else 0

    # Feature engineering com nomes atualizados
    input_data = np.array([[
        data.age,
        sex,
        data.years_of_education,
        data.apoe4,
        data.MMSE_mini_mental_state_examination,
        data.score_adas13,
        data.score_cdrsb,
        data.score_first_phase_RAVLT_test,
        data.score_forgetfulness_RAVLT,
        data.MMSE_mini_mental_state_examination / (data.score_adas13 + 1),
        data.age * data.apoe4,
        data.MMSE_mini_mental_state_examination ** 2,
        data.age ** 2
    ]])

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]

    dx_mapping = {
        0: "(CN) Cognitivamente Normal ",
        1: "(EMCI) Comprometimento Cognitivo Leve - Precoce",
        2: "(LMCI) Comprometimento Cognitivo Leve - Avançado ",
        3: "(AD) Doença de Alzheimer "
    }

    resultado = dx_mapping.get(pred, "Desconhecido")

    return {"diagnóstico_predito": resultado}
