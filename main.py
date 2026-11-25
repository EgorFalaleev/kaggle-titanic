from pathlib import Path

import pandas as pd

from config import config
from evaluation import create_pipeline, cross_validate_pipeline


def main():
    model_name = config.general.model

    print('Loading data')

    train = pd.read_csv(config.paths.train_data)
    test = pd.read_csv(config.paths.test_data)

    target = config.general.target

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.copy()

    print('Loading finished')

    print(f'Selected model: {model_name}')

    pipeline = create_pipeline(model_name, config)

    print(f'Cross-validation ({config.split.n_splits} folds)')

    cv_result = cross_validate_pipeline(pipeline, X_train, y_train, config)

    print(
        f'{config.general.metric} per fold:',
        ', '.join(f'{s:.4f}' for s in cv_result['scores'])
    )
    print(
        f"CV {config.general.metric}: {cv_result['mean']:.5f} "
        f"+/- {cv_result['std']:.5f}"
    )

    print('Starting final training on the whole train set')

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions.astype(int),
    })

    submission_path = Path(config.paths.submissions_dir) / f'{model_name}.csv'
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(submission_path, index=False)

    print(f"Training finished. Result saved to: {submission_path}")
    print(f"Predictions: {len(submission)}")
    print(submission.head())


if __name__ == '__main__':
    main()