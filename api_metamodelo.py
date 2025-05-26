from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F


scaler = joblib.load("transformers/scaler.pkl")
label_encoder = joblib.load("transformers/label_encoders.pkl")


rf_model = joblib.load('models/random_forest_model.pkl')
gb_model = joblib.load('models/gradient_boosting_model.pkl')
hist_gb_model = joblib.load('models/hist_gradient_boosting_model.pkl')
lgbm_model = joblib.load('models/lightgbm_model.pkl')
xgb_model = joblib.load('models/xgboost_model.pkl')
svm_model = joblib.load('models/svc_model.pkl')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(36, 128)   # Input layer
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 128)   # Hidden layer
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)   # Hidden layer
        self.dropout3 = nn.Dropout(0.2)
        self.out = nn.Linear(64, 5)# Output layer
    
    def forward(self, x):
        x = F.normalize(x)              # Optional: normalize inputs
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        return self.out(x)
    
net = Net()
net.load_state_dict(torch.load('models/neural_network_model.pth'))
net.eval()

model = joblib.load("models/metamodelo.pkl")

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
    
    ptgender = data.PTGENDER
    
    if ptgender == "Feminino":
        ptgender = "Female"
    elif ptgender == "Masculino":
        ptgender = "Male"
    
    fsversion_encoded = label_encoder["FSVERSION"].transform([data.FSVERSION])[0]        
    ptgender_encoded = label_encoder["PTGENDER"].transform([ptgender])[0]
    
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
        fsversion_encoded,
        data.Hippocampus,
        data.Entorhinal,
        data.ICV,
        data.mPACCdigit,
        data.mPACCtrailsB,
        data.FAQTOTAL,
        data.ADNI_MEM,
        ptgender_encoded
    ]])        
            
    input_scaled = scaler.transform(input_data)
    
    # Generate prediction probabilities scikit-learn models
    rf_probs = rf_model.predict_proba(input_scaled)
    gb_probs = gb_model.predict_proba(input_scaled)
    hgb_probs = hist_gb_model.predict_proba(input_scaled)
    lgbm_probs = lgbm_model.predict_proba(input_scaled)
    xgb_probs = xgb_model.predict_proba(input_scaled)
    svm_probs = svm_model.predict_proba(input_scaled)

    x_tensor = torch.from_numpy(input_scaled).float()
    with torch.no_grad():
        nn_logits = net(x_tensor)
        nn_probs = F.softmax(nn_logits, dim=1).numpy()
        
    stacked_features = np.hstack((rf_probs, gb_probs, hgb_probs, lgbm_probs, xgb_probs, svm_probs, nn_probs))
        
        
    pred = model.predict(stacked_features)[0]
        
    dx_mapping = {  
        0:  "(CN) Cognitivamente Normal ",
        1:  "(EMCI) Comprometimento Cognitivo Leve - Precoce",
        2:  "(LMCI) Comprometimento Cognitivo Leve - Avançado ",
        3:  "(SMC) Comprometimento Segnificativo da Memória",
        4:  "(AD) Doença de Alzheimer "
    }
    
    diagnosis = dx_mapping.get(pred)
    
    
    return {"diagnóstico_predito": diagnosis}
