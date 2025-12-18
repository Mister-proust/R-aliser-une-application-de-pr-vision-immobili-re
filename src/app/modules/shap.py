import shap
import pickle
import app.config as config

def load_shap():
    with open(config.MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    pipeline = bundle["pipeline"]
    X_shap_background = bundle["X_shap_background"]
    features = bundle["features"]

    xgb_model = pipeline.named_steps["model"]
    X_bg_transformed = pipeline.named_steps["pre"].transform(X_shap_background)

    explainer = shap.TreeExplainer(xgb_model, X_bg_transformed)

    return explainer, features, pipeline
