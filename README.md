# Student Performance Forecasting (Python, ML)

A complete, interview-ready project that predicts whether a student will **pass** an exam using
**Logistic Regression** and **Random Forest**. It includes model training, evaluation, feature importance,
and a simple **Streamlit** demo app.

## Project Structure
```
student_performance_forecasting/
  data/
    student_performance_sample.csv   # Sample dataset (replace with your real CSV if needed)
  models/
    best_model.joblib                # Saved after training
  reports/
    metrics.json                     # Metrics saved after training
    confusion_matrix.png             # Confusion matrix plot
    roc_curve.png                    # ROC curve plot
    feature_importances.csv          # Feature importances from RandomForest
  src/
    train_and_evaluate.py            # Main training + evaluation script
    app_streamlit.py                 # Streamlit demo to try predictions
requirements.txt
README.md
```

## Quickstart

1. (Optional) Create a virtual env, then install requirements:
```bash
pip install -r requirements.txt
```

2. Train and evaluate both models (LR and RF), save best model, and generate plots:
```bash
python src/train_and_evaluate.py --data data/student_performance_sample.csv --target passed
```

3. Launch the Streamlit demo:
```bash
streamlit run src/app_streamlit.py
```

## Replace with a real dataset
- Put your CSV into `data/` and run the training command with `--data path/to/your.csv` and `--target your_target_column`.
- The script automatically handles categorical vs numeric features using `OneHotEncoder` + `StandardScaler` inside pipelines.

## Notes
- Evaluation includes Accuracy, Precision, Recall, F1, ROC-AUC, CV accuracy, and plots (Confusion Matrix, ROC Curve).
- Best model is selected by validation ROC-AUC.
- Code is beginner-friendly and clean for interviews.
