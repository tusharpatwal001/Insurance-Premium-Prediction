from fastapi import FastAPI  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from schema.user_intput import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output, MODEL_VERSION, model
app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
def pridict_premium(data: UserInput):

    user_input = {
        "bmi": data.bmi,
        "age_group": data.age_group,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation,
    }

    try:
        prediction = predict_output(user_input)

        return JSONResponse(status_code=200, content={"response": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))


@app.get("/")
def home():
    return {"message": "Insurance Premium Prediction API"}


@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "version": MODEL_VERSION,
        "model_loaded": model is not None
    }
