from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI()

diabetes_ml_model = pickle.load(open('rf_model.pkl', 'rb'))

class DiabetesPredIn(BaseModel):
    embarazos: int
    glucosa: float
    presion_arterial: float
    espesor_piel: float
    insulina: float
    imc: float
    diabetes_pedigree_function: float
    edad: int

class DiabetesPredOut(BaseModel):
    tiene_diabetes: bool



@app.get('/')
def index():
    return {'mensaje': 'Diabetes APP'}


@app.post('/diabetes-predictions', response_model=DiabetesPredOut, status_code=201)
def procesar_prediccion_diabetes(diabetes_pred_in: DiabetesPredIn):
    
    input_values = [diabetes_pred_in.embarazos,
                    diabetes_pred_in.glucosa,
                    diabetes_pred_in.presion_arterial,
                    diabetes_pred_in.espesor_piel,
                    diabetes_pred_in.insulina,
                    diabetes_pred_in.imc,
                    diabetes_pred_in.diabetes_pedigree_function,
                    diabetes_pred_in.edad];
    
    print(input_values)
    
    features = [np.array(input_values)]
    print(features)
    
    features_df = pd.DataFrame(features)
    print(features_df)
    
    #Generamos las predicciones
    prediction_values = diabetes_ml_model.predict_proba(features_df)
    print(prediction_values)
    
    #Determinar la prediccion final
    final_prediction = np.argmax(prediction_values)
    print(final_prediction)
    
    return DiabetesPredOut(tiene_diabetes = final_prediction)
    
    
    


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)