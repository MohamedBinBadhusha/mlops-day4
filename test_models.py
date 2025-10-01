from pathlib import Path

svm_model_path = Path("svm_model.joblib")
dt_model_path = Path("dt_model.joblib")

assert svm_model_path.exists(), "SVM model file does not exist!"
assert dt_model_path.exists(), "Decision Tree model file does not exist!"

print("✅ Both model files exist!")

