# Titanic Survival Prediction

В данном репозитории представлено решени соревнования [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).

## Что реализовано

- **EDA:** анализ датасета, выявление пропусков и корреляций признаков, обоснования для feature engineering (`notebooks/01_eda.ipynb`).
- **Classic ML:** Logistic Regression (baseline), KNN, Decision Tree, Random Forest, XGBoost и CatBoost (`notebooks/02_baseline.ipynb`, `notebooks/03_models.ipynb`).
- **Оценка и подбор:** Кросс-валидация с помощью `StratifiedKFold`, подбор гиперпараметров через `GridSearchCV` (`notebooks/03_models.ipynb`).
- **Deep Learning:** MLP на PyTorch, эксперименты с BatchNorm и Dropout, Early Stopping (`notebooks/03_models.ipynb`).
- **Kaggle submission:** инференс на `test.csv` и сохранение файла для валидации решения на Kaggle (`main.py`).

## Установка и запуск

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Запуск инференса с формированием файла решения:

Для Classic ML моделей:
```bash
python main.py
```

Для Deep Learning моделей:
```bash
python main_dl.py
```

Файл с решением сохраняется в папку `submissions/`

## Результаты

Сравнение моделей проводится по средней Accuracy на стратифицированной кросс-валидации.

| Модель | CV Accuracy (mean) | Kaggle Accuracy |
|---|---|---|
| Logistic Regression (baseline) | 0.804271 | 0.77033 |
| CatBoost | 0.839548 | 0.77511 |
| XGBoost | 0.837935 | 0.76076 |
| Random Forest | 0.833135 | 0.77272 |
| KNN | 0.817110 | 0.73923 |
| Decision Tree | 0.815497 | 0.77511 |
| MLP (PyTorch) | — | 0.78229 |

