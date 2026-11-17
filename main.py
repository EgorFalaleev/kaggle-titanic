from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import config
from preprocessing import Preprocessor

from catboost import CatBoostClassifier


def main():
    print('Loading data')

    train = pd.read_csv(config.paths.train_data)
    test = pd.read_csv(config.paths.test_data)

    target = config.general.target

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.copy()

    print('Loading finished')

    model_params = OmegaConf.to_container(
        config.models.catboost.params,
        resolve=True
    )

    pipeline = Pipeline([
        ('preprocessing', Preprocessor()),
        ('model', CatBoostClassifier(**model_params)),
    ])

    print('Starting training')

    pipeline.fit(X_train, y_train)
    
    predictions = pipeline.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions.astype(int),
    })

    submission_path = Path(config.paths.submission)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(submission_path, index=False)

    print(f"Training finished. Result saved to: {submission_path}")
    print(f"Predictions: {len(submission)}")
    print(submission.head())


if __name__ == '__main__':
    main()
