import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

TITLE_MAPPING = {
    'Mlle': 'Miss',
    'Mme': 'Miss',
    'Ms': 'Miss',
    'Dr': 'Mr',
    'Major': 'Mr',
    'Lady': 'Mrs',
    'Countess': 'Mrs',
    'Jonkheer': 'Other',
    'Col': 'Other',
    'Rev': 'Other',
    'Capt': 'Mr',
    'Sir': 'Mr',
    'Don': 'Mr',
}

AGE_BY_TITLE = {
    'Mr': 33,
    'Mrs': 36,
    'Master': 5,
    'Miss': 22,
    'Other': 46,
}

SEX_MAPPING = {'male': 0, 'female': 1}
EMBARKED_MAPPING = {'S': 0, 'C': 1, 'Q': 2}
TITLE_ENCODING = {
    'Mr': 0,
    'Mrs': 1,
    'Miss': 2,
    'Master': 3,
    'Other': 4,
}


def handle_missing_values(data, age_median, fare_median):
    data = data.copy()

    data['Initial'] = data['Name'].str.extract(
        r'([A-Za-z]+)\.', expand=False
    )
    data['Initial'] = data['Initial'].replace(TITLE_MAPPING)
    data['Initial'] = data['Initial'].fillna('Other')

    data['Age'] = data['Age'].fillna(
        data['Initial'].map(AGE_BY_TITLE)
    )
    data['Age'] = data['Age'].fillna(age_median)

    data['Fare'] = data['Fare'].fillna(fare_median)
    data['Embarked'] = data['Embarked'].fillna('S')

    return data


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Split age and fares into groups and merge family related features"""
    data = data.copy()

    data['Age_group'] = pd.cut(
        data['Age'],
        bins=[-np.inf, 16, 32, 48, 64, np.inf],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(int)

    data['Family_size'] = data['Parch'] + data['SibSp']
    data['Alone'] = (data['Family_size'] == 0).astype(int)

    data['Fare_group'] = pd.cut(
        data['Fare'],
        bins=[-np.inf, 7.91, 14.454, 31, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    return data


def encode_features(data: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features as integers."""
    data = data.copy()

    data['Sex'] = data['Sex'].map(SEX_MAPPING)
    data['Embarked'] = data['Embarked'].map(EMBARKED_MAPPING)
    data['Initial'] = (
    data['Initial']
    .where(data['Initial'].isin(TITLE_ENCODING), 'Other')
    .map(TITLE_ENCODING)
)

    return data


def drop_features(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'Name',
        'Age',
        'Ticket',
        'Fare',
        'Cabin',
        'PassengerId',
    ]

    return data.drop(columns=columns_to_drop, errors='ignore')


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = handle_missing_values(data)
    data = create_features(data)
    data = encode_features(data)
    data = drop_features(data)

    return data

class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.age_median = X['Age'].median()
        self.fare_median = X['Fare'].median()

        return self

    def transform(self, X):
        X = handle_missing_values(
            X,
            age_median=self.age_median,
            fare_median=self.fare_median
        )

        X = create_features(X)
        X = encode_features(X)
        X = drop_features(X)

        return X