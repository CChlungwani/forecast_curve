from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_INPUT_DIR = PROJECT_ROOT / "data_input"
FEATURES_DIR = PROJECT_ROOT / "features"
LABEL_CREATION_DIR = PROJECT_ROOT / "label_creation"
MODEL_TRAINING_DIR = PROJECT_ROOT / "model_training"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURE_IMPORTANCE_DIR = PROJECT_ROOT / "feature_importance"