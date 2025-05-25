from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

scaler = joblib.load("transformers/scaler.pkl")
label_encoder = joblib.load("transformers/label_encoders.pkl")
model = joblib.load("metamodelo.pkl")

api = FastAPI()


class AlzheimerInput(BaseModel):
    
    
    AGE: float
    PTEDUCAT: int
    FDG: float
    AV45: float
    CDRSB: float
    ADAS11: float
    ADAS13: float
    ADASQ4: float
    MMSE: float
    RAVLT_immediate: float
    RAVLT_perc_forgetting: float
    LDELTOTAL: float
    TRABSCOR: float
    FAQ: float
    MOCA: float
    EcogPtMem: float
    EcogPtLang: float
    EcogPtVisspat: float
    EcogPtDivatt: float
    EcogPtTotal: float
    EcogSPMem: float
    EcogSPLang: float
    EcogSPVisspat: float
    EcogSPPlan: float
    EcogSPOrgan: float
    EcogSPDivatt: float
    EcogSPTotal: float
    FSVERSION: str
    Hippocampus: float
    Entorhinal: float
    ICV: float
    mPACCdigit: float
    mPACCtrailsB: float
    FAQTOTAL: float
    ADNI_MEM: float
    PTGENDER: str


app = FastAPI()

@app.post("/predict")
def predict(data: AlzheimerInput):
    
    ptgender = "Male" if data.PTGENDER == "Masculino" else "Female"
    
    ptgender_encoded = label_encoder["PTGENDER"].transform([ptgender])[0]
    fsversion_encoded = label_encoder["FSVERSION"].transform([data.FSVERSION])[0]

    categorical_encoded = np.array([[ptgender_encoded, fsversion_encoded]])
        
    
    input_data = np.array([[
        
        data.AGE,
        data.PTEDUCAT,
        data.FDG,
        data.AV45,
        data.CDRSB,
        data.ADAS11,
        data.ADAS13,
        data.ADASQ4,
        data.MMSE,
        data.RAVLT_immediate,
        data.RAVLT_perc_forgetting,
        data.LDELTOTAL,
        data.TRABSCOR,
        data.FAQ,
        data.MOCA,
        data.EcogPtMem,
        data.EcogPtLang,
        data.EcogPtVisspat,
        data.EcogPtDivatt,
        data.EcogPtTotal,
        data.EcogSPMem,
        data.EcogSPLang,
        data.EcogSPVisspat,
        data.EcogSPPlan,
        data.EcogSPOrgan,
        data.EcogSPDivatt,
        data.EcogSPTotal,
        data.Hippocampus,
        data.Entorhinal,
        data.ICV,
        data.mPACCdigit,
        data.mPACCtrailsB,
        data.FAQTOTAL,
        data.ADNI_MEM,
    ]])        
    
    input_data = np.concatenate((input_data, categorical_encoded), axis=1)  
        
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]

    
    dx_mapping = {  
        0:  "(CN) Cognitivamente Normal ",
        1:  "(EMCI) Comprometimento Cognitivo Leve - Precoce",
        2:  "(LMCI) Comprometimento Cognitivo Leve - Avançado ",
        3:  "(SMC) Comprometimento Segnificativo da Memória",
        4:  "(AD) Doença de Alzheimer "
    }
    
    diagnosis = dx_mapping.get(pred)
    
    
    
    
    
    
    return {"diagnóstico_predito": diagnosis}
