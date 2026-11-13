from pathlib import Path
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parent

config = {
    'general': {
        'seed': 42,
        'target': 'Survived',
        'metric': 'accuracy',
    },

    'paths': {
        'train_data': str(BASE_DIR / 'data' / 'train.csv'),
        'test_data': str(BASE_DIR / 'data' / 'test.csv'),
    },

    'split': {
        'test_size': 0.3,
        'n_splits': 10,
        'shuffle': True,
        'random_state': '${general.seed}',
    },

    
    'models': {
        'logistic_regression': {
            'enabled': True,
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': '${general.seed}',
            },
        },

        'knn': {
            'enabled': True,
            'params': {
                'n_neighbors': 10,
                'weights': 'uniform',
                'metric': 'minkowski',
            },
        },

        'decision_tree': {
            'enabled': True,
            'params': {
                'max_depth': 5,
                'min_samples_split': 2,
                'random_state': '${general.seed}',
            },
        },

        'random_forest': {
            'enabled': True,
            'params': {
                'n_estimators': 100,
                'max_depth': 4,
                'random_state': '${general.seed}',
                'n_jobs': -1,
            },
        },

        'xgboost': {
            'enabled': True,
            'params': {
                'n_estimators': 1000,
                'learning_rate': 0.001,
                'max_depth': 4,
                'min_child_weight': 2,
                'gamma': 0.9,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'random_state': '${general.seed}',
                'n_jobs': -1,
            },
        },

        'catboost': {
            'enabled': True,
            'params': {
                'iterations': 1000,
                'learning_rate': 0.001,
                'depth': 4,
                'loss_function': 'Logloss',
                'verbose': False,
                'random_seed': '${general.seed}',
                'thread_count': -1,
            },
        },
    },
}

config = OmegaConf.create(config)
