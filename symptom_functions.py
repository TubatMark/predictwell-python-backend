import sys
import pandas as pd
import joblib
import json
import os
import logging

# Configure logging
logging.basicConfig(
    filename="diagnosis_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Path to the saved RandomForest model
MODEL_PATH = "random_forest_classifier_disease_model.pkl"


def load_symptoms():
    """Load symptoms from a text file into a set."""
    try:
        # Construct the absolute path to symptoms.txt
        file_path = os.path.join(os.path.dirname(__file__), "symptoms.txt")
        with open(file_path, "r") as f:
            symptoms = {line.strip().lower() for line in f if line.strip()}
        return symptoms
    except FileNotFoundError:
        return set()


def load_disease_data():
    """Load disease data from an Excel file into a DataFrame."""
    try:
        df = pd.read_excel("disease_dataset.xlsx")
        logging.info("Disease data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error("Disease dataset file not found: disease_dataset.xlsx")
        return None


def load_model():
    """Load the pre-trained model and feature names."""
    try:
        # Construct the absolute path to the model file
        model_path = os.path.join(
            os.path.dirname(__file__), "random_forest_classifier_disease_model.pkl"
        )
        logging.info(f"Attempting to load model from: {model_path}")
        model, feature_names = joblib.load(model_path)
        logging.info(f"Model loaded successfully. Features: {feature_names}")
        return model, feature_names
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        return None, []
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, []


def preprocess_symptoms(user_input):
    """Extract and match symptoms from user input."""
    symptoms_in_file = load_symptoms()
    if not symptoms_in_file:
        return {"error": "Symptoms file not found or empty."}

    # Normalize user input and match symptoms
    input_symptoms = {
        symptom.strip().replace("_", " ").lower()
        for symptom in user_input.replace(",", " ").split()
    }
    matched_symptoms = [
        symptom for symptom in input_symptoms if symptom in symptoms_in_file
    ]

    logging.debug(f"Preprocessed symptoms: {matched_symptoms}")
    return {"matched_symptoms": matched_symptoms}


def create_feature_vector(matched_symptoms, feature_names):
    """Creates a feature vector DataFrame based on matched symptoms."""
    feature_vector = [
        1 if symptom in matched_symptoms else 0 for symptom in feature_names
    ]
    feature_vector_df = pd.DataFrame([feature_vector], columns=feature_names)
    logging.debug(f"Feature vector created: {feature_vector}")
    return feature_vector_df


def diagnose_symptoms(matched_symptoms):
    """Diagnose based on matched symptoms using a RandomForestClassifier model."""
    model, feature_names = load_model()
    if model is None or not feature_names:
        return {"error": "Model file not found or invalid."}

    if not matched_symptoms:
        return {"prognosis": "No matching symptoms found."}

    try:
        feature_vector_df = create_feature_vector(matched_symptoms, feature_names)

        # Debug the feature vector
        logging.debug(f"Feature vector for prediction: {feature_vector_df}")

        prediction = model.predict(feature_vector_df)[0]
        probabilities = model.predict_proba(feature_vector_df)[0]
        probability = max(probabilities) * 100  # Probability of the predicted class

        logging.info(f"Diagnosis: {prediction}, Probability: {probability}%")
        return {"prognosis": prediction, "probability": probability}
    except Exception as e:
        logging.error(f"Error during diagnosis: {str(e)}")
        return {"error": "An error occurred during diagnosis."}


if __name__ == "__main__":
    try:
        user_input = sys.argv[1] if len(sys.argv) > 1 else ""

        # Default response structure
        result = {
            "matched_symptoms": [],
            "prognosis": "No input provided.",
            "probability": "N/A",
        }

        if not user_input.strip():
            result["error"] = "No symptoms provided."
        else:
            # Preprocess user input
            preprocessed_result = preprocess_symptoms(user_input)
            if "error" in preprocessed_result:
                result.update(preprocessed_result)
            else:
                matched_symptoms = preprocessed_result["matched_symptoms"]
                result.update(preprocessed_result)

                if matched_symptoms:
                    diagnosis_result = diagnose_symptoms(matched_symptoms)
                    result["prognosis"] = diagnosis_result.get(
                        "prognosis", "No likely diagnosis found."
                    )
                    result["probability"] = diagnosis_result.get("probability", "N/A")
                else:
                    result["prognosis"] = "No matching symptoms found."

        print(json.dumps(result))
        sys.stdout.flush()
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        print(json.dumps({"error": f"An unexpected error occurred: {str(e)}"}))
        sys.stdout.flush()
