from pathlib import Path
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parent

config = {
    'general': {
        'experiment_name' : 'catboost',
        'seed': 42,
        'target': 'Survived',
        'metric': 'accuracy',
    },

    'paths': {
        'train_data': str(BASE_DIR / 'data' / 'train.csv'),
        'test_data': str(BASE_DIR / 'data' / 'test.csv'),
        'submission': (
            BASE_DIR / 'submissions' / '${general.experiment_name}.csv'
        ).as_posix(),
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
                'n_estimators': 300,
                'max_depth': 5,
                'min_samples_leaf': 4,
                'random_state': '${general.seed}',
                'n_jobs': -1,
            },
        },

        'xgboost': {
            'enabled': True,
            'params': {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 4,
                'min_child_weight': 3,
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
                'iterations': 500,
                'learning_rate': 0.01,
                'depth': 4,
                'loss_function': 'Logloss',
                'verbose': False,
                'random_seed': '${general.seed}',
                'thread_count': -1,
                'l2_leaf_reg': 3
            },
        },
    },
        
    'dl': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'use_batchnorm': False,
        'use_dropout': True,
        'dropout_rate': 0.2,
    },

    'tuning': {
        'n_splits': 5,
        'scoring': 'accuracy',
        'n_jobs': -1,

        'param_grids': {
            'logistic_regression': {
                'model__C': [0.01, 0.1, 1, 10, 100],
            },

            'knn': {
                'model__n_neighbors': [3, 5, 7, 9, 11, 15],
                'model__weights': ['uniform', 'distance'],
                'model__p': [1, 2],
            },

            'decision_tree': {
                'model__max_depth': [3, 5, 7],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
            },

            'random_forest': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7],
                'model__min_samples_leaf': [1, 2, 4],
            },

            'xgboost': {
                'model__n_estimators': [200, 500, 1000],
                'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
                'model__max_depth': [2, 3, 4],
                'model__min_child_weight': [1, 3],
            },

            'catboost': {
                'model__iterations': [300, 500, 1000],
                'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
                'model__depth': [3, 4, 5],
                'model__l2_leaf_reg': [1, 3],
            },
        },
    },
}

config = OmegaConf.create(config)
