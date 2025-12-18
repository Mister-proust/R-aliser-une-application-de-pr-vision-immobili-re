from app.modules.shap import load_shap

explainer = None
features = None
pipeline = None

def init_state():
    global explainer, features, pipeline
    explainer, features, pipeline = load_shap()
