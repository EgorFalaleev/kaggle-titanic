
import pandas as pd

from omegaconf import OmegaConf
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from evaluation import create_pipeline


def tune_model(name, X, y, config):
    pipeline = create_pipeline(name, config)

    param_grid = OmegaConf.to_container(
        config.tuning.param_grids[name],
        resolve=True
    )

    cv = StratifiedKFold(
        n_splits=config.tuning.n_splits,
        shuffle=True,
        random_state=config.general.seed
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=config.tuning.scoring,
        cv=cv,
        n_jobs=config.tuning.n_jobs,
        refit=True,
        verbose=1
    )

    search.fit(X, y)

    return search


def tune_all_models(X, y, config):
    results = []
    best_models = {}

    for name, model_config in config.models.items():
        if not model_config.enabled:
            continue

        print(f"\nTuning {name}...")

        search = tune_model(name, X, y, config)

        results.append({
            'Model': name,
            'CV Mean': search.best_score_,
            'CV Std': search.cv_results_['std_test_score'][
                search.best_index_
            ],
            'Best Params': search.best_params_
        })

        best_models[name] = search.best_estimator_

    results_df = (
        pd.DataFrame(results)
        .sort_values('CV Mean', ascending=False)
        .reset_index(drop=True)
    )

    return results_df, best_models
