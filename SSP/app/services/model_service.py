import os

import joblib
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(APP_DIR, "models")

OUTDOOR_MODEL_PATH = os.path.join(MODEL_DIR, "best_pothole_model.pkl")
INDOOR_MODEL_PATH = os.path.join(MODEL_DIR, "best_surface_model.pkl")

OUTDOOR_MODEL = joblib.load(OUTDOOR_MODEL_PATH)
INDOOR_MODEL = joblib.load(INDOOR_MODEL_PATH)

OUTDOOR_FEATURES = [str(name) for name in OUTDOOR_MODEL.feature_names_in_]
INDOOR_FEATURES = [str(name) for name in INDOOR_MODEL.feature_names_in_]


def _make_frame(row_dict, feature_names):
    data = {name: [row_dict[name]] for name in feature_names}
    return pd.DataFrame(data, columns=feature_names)


def predict_outdoor(row_dict):
    frame = _make_frame(row_dict, OUTDOOR_FEATURES)
    pothole_prob = float(OUTDOOR_MODEL.predict_proba(frame)[0][1])
    pred_idx = int(OUTDOOR_MODEL.predict(frame)[0])
    pred_label = "pothole" if pred_idx == 1 else "normal_road"
    pred_prob = pothole_prob if pred_idx == 1 else 1.0 - pothole_prob
    return {
        "pred_label": pred_label,
        "pred_prob": pred_prob,
        "raw_prob": pothole_prob,
        "model_name": "best_pothole_model.pkl",
        "feature_table_type": "outdoor",
    }


def predict_indoor(row_dict, label_map):
    frame = _make_frame(row_dict, INDOOR_FEATURES)
    probs = INDOOR_MODEL.predict_proba(frame)[0]
    best_idx = int(probs.argmax())
    class_value = INDOOR_MODEL.classes_[best_idx]
    class_key = int(class_value) if str(class_value).lstrip("-").isdigit() else class_value
    pred_label = label_map.get(class_key, str(class_value))
    pred_prob = float(probs[best_idx])
    return {
        "pred_label": pred_label,
        "pred_prob": pred_prob,
        "raw_prob": pred_prob,
        "model_name": "best_surface_model.pkl",
        "feature_table_type": "indoor",
    }
