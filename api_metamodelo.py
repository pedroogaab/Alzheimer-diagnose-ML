from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

scaler = joblib.load("transformers/scaler_alzheimer.pkl")
label_encoder = joblib.load("transformers/label_encoders.pkl")
model = joblib.load("metamodelo.pkl")

api = FastAPI()
class AlzheimerInput(BaseModel):
    AGE: float
    PTEDUCAT: int
    FDG: float
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
    EcogPtDivatt: float
    EcogPtTotal: float
    EcogSPMem: float
    EcogSPLang: float
    EcogSPVisspat: float
    EcogSPPlan: float
    EcogSPOrgan: float
    EcogSPDivatt: float
    EcogSPTotal: float
    FLDSTRENG: str
    FSVERSION: str
    IMAGEUID: float
    Hippocampus: float
    mPACCdigit: float
    mPACCtrailsB: float
    FAQTOTAL: float
    ID_faq: float
    ID_neurobat: float
    ADNI_MEM: float


app = FastAPI()

@app.post("/predict")
def predict(data: AlzheimerInput):
    input_data = np.array([[
        data.AGE,
        data.PTEDUCAT,
        data.FDG,
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
        data.EcogPtDivatt,
        data.EcogPtTotal,
        data.EcogSPMem,
        data.EcogSPLang,
        data.EcogSPVisspat,
        data.EcogSPPlan,
        data.EcogSPOrgan,
        data.EcogSPDivatt,
        data.EcogSPTotal,
        data.FLDSTRENG,
        data.FSVERSION,
        data.IMAGEUID,
        data.Hippocampus,
        data.mPACCdigit,
        data.mPACCtrailsB,
        data.FAQTOTAL,
        data.ID_faq,
        data.ID_neurobat,
        data.ADNI_MEM,
    ]])        
    
    dx_mapping = {  
        "CN":   "(CN) Cognitivamente Normal ",
        "EMCI": "(EMCI) Comprometimento Cognitivo Leve - Precoce",
        "LMCI": "(LMCI) Comprometimento Cognitivo Leve - Avançado ",
        "SMC":  "(SMC) Comprometimento Segnificativo da Memória",
        "AD":   "(AD) Doença de Alzheimer "
    }