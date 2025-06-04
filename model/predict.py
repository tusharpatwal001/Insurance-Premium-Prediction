import pickle
import pandas as pd

# import the ml model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# mlflow
MODEL_VERSION = "1.0.0"

# get class labels from model (important for matching probabilities to class names)
class_labels = model.classes_.tolist()


def predict_output(user_input: dict):

    df = pd.DataFrame([user_input])

    # predict the class
    predicted_class = model.predict(df)[0]

    # get probabilities for all classes
    probalities = model.predict_proba(df)[0]
    confidence = max(probalities)

    # create mapping: {class_name: probality}
    class_probs = dict(zip(class_labels, map(
        lambda p: round(p, 4), probalities)))

    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probalities": class_probs
    }
