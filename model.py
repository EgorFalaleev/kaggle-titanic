
from omegaconf import OmegaConf

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


MODEL_MAP = {
    'logistic_regression': LogisticRegression,
    'knn': KNeighborsClassifier,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'xgboost': XGBClassifier,
    'catboost': CatBoostClassifier,
}


def create_model(name, config):
    model_class = MODEL_MAP[name]

    params = OmegaConf.to_container(
        config.models[name].params,
        resolve=True
    )

    return model_class(**params)
