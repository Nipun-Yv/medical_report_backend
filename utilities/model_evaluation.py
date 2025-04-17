import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from models.generalisedANN import ANNClassifier
from models.diabetesANN import DiabetesClassifier

import numpy as np
import torch.nn.functional as F

emb_szs=[(3, 1), (6, 3)]
model1=DiabetesClassifier(emb_szs,6)
model2=ANNClassifier(11,2,[100,100])

model1.load_state_dict(torch.load('./stored_weights/diabetes_model_weights.pth'))
model2.load_state_dict(torch.load('./stored_weights/cardiovascular_diseases_weights.pth'))

# scaler1:MinMaxScaler = joblib.load('./scalers/scaler1.pkl')
scaler2:MinMaxScaler = joblib.load('./scalers/scaler2.pkl')

def generate_confidence_score(diabetes_sample:list,card_sample:list):

    X_cont=np.array(diabetes_sample[0:6]).reshape(1,-1)
    X_cat=np.array(diabetes_sample[6:]).reshape(1,-1)
    X_eval2=np.array(card_sample).reshape(1,-1)

    X_cont=torch.tensor(X_cont,dtype=torch.float)
    X_cat=torch.tensor(X_cat,dtype=torch.long)
    X_eval2=torch.FloatTensor(scaler2.transform(X_eval2))

    model1.eval()
    model2.eval()

    print(X_cont,X_cat)
    print(X_eval2)
    y_pred=model1(X_cont,X_cat)
    y_pred2=model2(X_eval2)

    print(y_pred,y_pred2)
    y_pred=F.softmax(y_pred).tolist()
    y_pred2=F.softmax(y_pred2).tolist()
    return [y_pred,y_pred2]

