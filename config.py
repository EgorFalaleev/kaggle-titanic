from pathlib import Path
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parent

config = {
    'general': {
        'seed': 42,
        'target': 'Survived',
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

    'logreg': {
        'max_iter': 1000,
        'random_state': '${general.seed}',
    },
}

config = OmegaConf.create(config)
