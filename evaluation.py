
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import Preprocessor
from model import create_model


SCALED_MODELS = {'logistic_regression', 'knn'}


def create_pipeline(name, config):
    steps = [
        ('preprocessing', Preprocessor()),
    ]

    if name in SCALED_MODELS:
        steps.append(('scaler', StandardScaler()))

    steps.append(('model', create_model(name, config)))

    return Pipeline(steps)


def compare_models(X, y, config):
    cv = StratifiedKFold(
        n_splits=config.split.n_splits,
        shuffle=config.split.shuffle,
        random_state=config.general.seed,
    )

    results = []

    for name, model_config in config.models.items():
        if not model_config.enabled:
            continue

        pipeline = create_pipeline(name, config)

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=config.general.metric,
            n_jobs=1,
        )

        results.append({
            'Model': name,
            'CV Mean': scores.mean(),
            'CV Std': scores.std(),
        })

    return (
        pd.DataFrame(results)
        .sort_values('CV Mean', ascending=False)
        .reset_index(drop=True)
    )
