import os
import logging
from flask import Flask, request, jsonify, render_template
from symptom_functions import (
    preprocess_symptoms,
    diagnose_symptoms,
)  # Import functions from symptom_functions.py

app = Flask(__name__)
# testing
# Logging setup
# Commented out to avoid issues in read-only environments
# log_file = "app.log"  # Log file name
# if not os.path.exists(log_file):
#     with open(log_file, "w") as file:  # Create the log file if it doesn't exist
#         file.write("Application Log\n")

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all types of logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Log to console instead of a file
)


@app.route("/home")
def home():
    logging.info("Accessed /home route")
    return render_template("index.html")


@app.route("/preprocess", methods=["POST"])
def preprocess_route():
    """
    Route for preprocessing symptoms without running a diagnosis.
    """
    try:
        # Parse JSON input
        data = request.json
        user_input = data.get("message", "")

        if not user_input.strip():
            error_message = "No symptoms provided."
            logging.warning(error_message)
            return jsonify({"error": error_message}), 400

        logging.info(f"Received input for preprocessing: {user_input}")

        # Preprocess symptoms
        preprocessed_result = preprocess_symptoms(user_input)
        if "error" in preprocessed_result:
            logging.error(f"Error during preprocessing: {preprocessed_result}")
            return jsonify(preprocessed_result), 500

        logging.info(f"Preprocessed symptoms: {preprocessed_result}")
        return jsonify(preprocessed_result)
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 500


@app.route("/process", methods=["POST"])
def process_symptoms():
    try:
        # Parse JSON input
        data = request.json
        user_input = data.get("message", "")

        if not user_input.strip():
            error_message = "No symptoms provided."
            logging.warning(error_message)
            return jsonify({"error": error_message}), 400

        logging.info(f"Received input: {user_input}")

        # Step 1: Preprocess symptoms
        preprocessed_result = preprocess_symptoms(user_input)
        if "error" in preprocessed_result:
            logging.error(f"Error during preprocessing: {preprocessed_result}")
            return jsonify(preprocessed_result), 500

        matched_symptoms = preprocessed_result.get("matched_symptoms", [])

        # Step 2: Use diagnose_symptoms to process matched symptoms
        if matched_symptoms:
            logging.info(f"Matched symptoms: {matched_symptoms}")
            diagnosis_result = diagnose_symptoms(matched_symptoms)
            response = {
                "matched_symptoms": matched_symptoms,
                "prognosis": diagnosis_result.get(
                    "prognosis", "No likely diagnosis found."
                ),
                "probability": diagnosis_result.get("probability", "N/A"),
            }
        else:
            logging.info("No matching symptoms found.")
            response = {
                "matched_symptoms": [],
                "prognosis": "No matching symptoms found.",
                "probability": "N/A",
            }

        logging.info(f"Response sent: {response}")
        return jsonify(response)

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 500


if __name__ == "__main__":
    logging.info("Starting Flask app")
    app.run(debug=False)
