import joblib

model = joblib.load("extratree_model.pkl")
encoder = {
    col: joblib.load(f"{col}_encoder.pkl") 
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}
print(model.feature_names_in_)
print(encoder["Sex"].classes_)
print(encoder["Housing"].classes_)
print(encoder["Saving accounts"].classes_)
print(encoder["Checking account"].classes_)
