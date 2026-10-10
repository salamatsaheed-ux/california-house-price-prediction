# california-house-price-prediction

Predicting median house values in California with scikit-learn. The project starts with a linear regression baseline, then shows how switching to a random forest and tuning it with `GridSearchCV` changes the results.

## Dataset

[California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), loaded directly through `sklearn.datasets.fetch_california_housing`. It has about 20,000 districts and 8 features:

| Feature | Meaning |
|---|---|
| `MedInc` | Median income in the district |
| `HouseAge` | Median house age |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | District population |
| `AveOccup` | Average household occupancy |
| `Latitude` | District latitude |
| `Longitude` | District longitude |

**Target:** median house value, in units of $100,000.

## Approach

1. Split the data 75% train / 25% test.
2. **Baseline:** `StandardScaler` + `LinearRegression` in a `Pipeline`.
3. **Random forest:** `RandomForestRegressor` with default settings. The scaler was removed here because tree models split on thresholds and are not affected by feature scale.
4. **Tuning:** `GridSearchCV` (5-fold CV, scored on negative MSE) over `n_estimators`, `max_depth`, `min_samples_split` and `min_samples_leaf`.
5. A small `predict_house_price()` helper turns raw feature values into a dollar estimate.

## Results (test set)

| Model | MSE | R² |
|---|---|---|
| Linear regression | 0.54 | 0.611 |
| Random forest (default) | 0.266 | 0.807 |
| Random forest (tuned) | 0.262 | 0.810 |

**Best parameters:** `n_estimators=200`, `max_depth=20`, `min_samples_split=2`, `min_samples_leaf=1`.

## Key takeaways

- Moving from linear regression to a random forest roughly halved the error. The data contains non-linear patterns (especially around location) that a straight line cannot capture.
- Hyperparameter tuning gave only a small further gain. Choosing the right model type mattered far more than tuning.
- The cross-validation MSE (0.263) is close to the test MSE (0.262), so the tuned model is not overfitting.
- Feature scaling matters for linear regression but not for tree-based models.

## Example prediction

```python
predict_house_price(
    my_model=california,
    MedInc=8.3, HouseAge=41.0, AveRooms=6.0, AveBedrms=1.0,
    Population=980, AveOccup=2.5, Latitude=37.5, Longitude=-122.3
)
```

## Run it yourself

```bash
git clone https://github.com/salamatsaheed-ux/california-house-price-prediction.git
cd california-house-price-prediction
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook House-Price-Prediction.ipynb
```

## Possible next steps

- Try gradient boosting (`HistGradientBoostingRegressor`) and tune `learning_rate` and `max_depth`.
- Add visualizations of feature importance and prediction errors.

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter Notebook
