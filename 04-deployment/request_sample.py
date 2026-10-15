import requests


url = "http://localhost:7676/predict"

sample_customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "fiber_optic",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "yes",
    "streamingmovies": "no",
    "contract": "one_year",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 36,
    "monthlycharges": 100.67,
    "totalcharges": 3080.5
}

response = requests.post(url, json=sample_customer)
result = response.json()

churn_proba = result.get("churn_probability")

if churn_proba:
    print(f"Probability of Churn: {churn_proba:.4f}")

    if churn_proba >= 0.5:
        print("Send promotional campaign.")
    else:
        print("Do nothing.")
else:
    print(result)
