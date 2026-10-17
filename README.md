# FocusFlow

ML-based personal productivity and day planner. It learns how sleep, phone use, stress, energy and
motivation relate to productivity, predicts productivity and distraction risk for each hour, and
ranks the best focus hours.

Navvinram R

## Run

```bash
pip install -r requirements.txt
streamlit run app.py        # opens http://localhost:8501
```

On first run the app trains the models on synthetic data and saves `focusflow_models.joblib`.
To train from the terminal and print the metrics: `python train_models.py`.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: check-in sliders, model comparison, hourly forecast, best focus hours |
| `train_models.py` | Trains 3 regressors and 3 classifiers, reports held-out metrics |
| `scheduler.py` | Hourly forecast, focus-score ranking, greedy task assignment |
| `data_generator.py` | Synthetic dataset (users x days x hours) |
| `db.py` | SQLite schema and check-in helper |

## Notes

- Data is synthetic, so scores are not results for real users.
- This is a planning aid, not a medical tool.
