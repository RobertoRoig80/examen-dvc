"""
Evaluation: calcul des métriques et sauvegarde des prédictions
Usage: python src/models/evaluate.py --input_dir examen_dvc/data/processed --model examen_dvc/models/model.pkl --metrics_dir examen_dvc/metrics --out_csv examen_dvc/data/predictions.csv
"""
import argparse
import os
import joblib
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def main(input_dir, model_path, metrics_dir, out_csv):
    scaler = joblib.load(os.path.join(input_dir, 'scaler.pkl'))
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv'))
    if y_test.shape[1] > 1:
        y_test = y_test.iloc[:, 0]

    model = joblib.load(model_path)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    os.makedirs(metrics_dir, exist_ok=True)
    metrics = {'mse': float(mse), 'mae': float(mae), 'r2': float(r2)}
    with open(os.path.join(metrics_dir, 'scores.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # sauvegarde des predictions
    df_preds = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
    df_preds['y_test'] = y_test.values
    df_preds['y_pred'] = preds
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_preds.to_csv(out_csv, index=False)
    print('Metrics:', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data/processed')
    parser.add_argument('--model', type=str, default='./models/model.pkl')
    parser.add_argument('--metrics_dir', type=str, default='./metrics')
    parser.add_argument('--out_csv', type=str, default='./data/predictions.csv')
    args = parser.parse_args()
    main(args.input_dir, args.model, args.metrics_dir, args.out_csv)