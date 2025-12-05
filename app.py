from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle et le scaler déjà entraînés
model = joblib.load("model.pkl")      # modèle RandomForest ou autre
scaler = joblib.load("scaler.pkl")    # StandardScaler

# Définir les colonnes/features manuellement (d'après diabetes.csv)
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST","GET"])
def predict():
    try:
        # Récupérer les valeurs saisies dans le formulaire
        features = [float(request.form[col]) for col in columns]
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        result = "Diabète détecté" if prediction == 1 else "Pas de diabète"
    except Exception as e:
        result = f"Erreur : {e}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
