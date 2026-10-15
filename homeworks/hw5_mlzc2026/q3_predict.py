import pickle


with open("pipeline.bin", "rb") as f_in:
    model_pipeline = pickle.load(f_in)


sample_lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}


convert_proba = model_pipeline.predict_proba(sample_lead)[0]
print(model_pipeline.classes_)
print(convert_proba)

print(f"Probability of conversion: {convert_proba[-1]:.3f}")
