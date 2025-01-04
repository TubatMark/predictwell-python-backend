import pandas as pd


def extract_symptoms_to_txt():
    try:
        # Load the Excel file
        df = pd.read_excel("disease_dataset.xlsx")

        # Get the symptom columns and save them to a text file
        symptoms = [
            col.lower() for col in df.columns[:-1]
        ]  # Exclude the last column (diagnosis)

        # Save symptoms to a text file
        with open("symptoms.txt", "w") as f:
            for symptom in symptoms:
                f.write(f"{symptom}\n")

        print("Symptoms successfully extracted to symptoms.txt")
    except FileNotFoundError:
        print("Error: 'disease_dataset.xlsx' file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Run the function
extract_symptoms_to_txt()
