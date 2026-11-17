from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from config import config

import torch
from dl import prepare_data, MLP, train_model


def main():
    print('Loading data')

    train = pd.read_csv(config.paths.train_data)
    test = pd.read_csv(config.paths.test_data)

    target = config.general.target

    X = train.drop(columns=[target])
    y = train[target]
    X_test = test.copy()

    print('Loading finished')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.split.test_size,
        random_state=config.general.seed,
        stratify=y
    )

    train_dataset, val_dataset, preprocessor, scaler = prepare_data(X_train, X_val, y_train, y_val)

    X_test = preprocessor.transform(test)
    X_test = scaler.transform(X_test).astype(np.float32)

    print('Starting training')

    model = MLP(
        input_dim=train_dataset.tensors[0].shape[1],
        use_batchnorm=config.dl.use_batchnorm,
        use_dropout=config.dl.use_dropout,
        dropout_rate=config.dl.dropout_rate
    ).to(device)

    best_model_state, _, _ = train_model(model, train_dataset, val_dataset, device)

    model.load_state_dict(best_model_state)

    model.eval()

    X_test_tensor = torch.tensor(
        X_test, dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        predictions = (logits >= 0).int().cpu().numpy()
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
