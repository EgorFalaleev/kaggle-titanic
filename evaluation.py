import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import Preprocessor
from model import create_model


SCALED_MODELS = {'logistic_regression', 'knn'}


def create_pipeline(name, config):
    """Build the full pipeline (Preprocessor [+ scaler] + model) by model name.

    Scaling is added only for models that need it (``SCALED_MODELS``).
    """
    steps = [
        ('preprocessing', Preprocessor()),
    ]

    if name in SCALED_MODELS:
        steps.append(('scaler', StandardScaler()))

    steps.append(('model', create_model(name, config)))

    return Pipeline(steps)


def make_cv(config):
    return StratifiedKFold(
        n_splits=config.split.n_splits,
        shuffle=config.split.shuffle,
        random_state=config.general.seed,
    )


def cross_validate_pipeline(pipeline, X, y, config):
    """Cross-validate a full pipeline.

    Returns the per-fold scores and their mean and std for
    ``config.general.metric``.
    """
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=make_cv(config),
        scoring=config.general.metric,
        n_jobs=1,
    )

    return {
        'scores': scores,
        'mean': float(scores.mean()),
        'std': float(scores.std()),
    }


def compare_models(X, y, config):
    """Cross-validate every enabled model; return a table sorted by mean score."""
    results = []

    for name, model_config in config.models.items():
        if not model_config.enabled:
            continue

        pipeline = create_pipeline(name, config)
        cv_result = cross_validate_pipeline(pipeline, X, y, config)

        results.append({
            'Model': name,
            'CV Mean': cv_result['mean'],
            'CV Std': cv_result['std'],
        })

    return (
        pd.DataFrame(results)
        .sort_values('CV Mean', ascending=False)
        .reset_index(drop=True)
    )