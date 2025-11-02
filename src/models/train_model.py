"""
Entraînement du modèle en utilisant les paramètres sauvegardés
Usage: python src/models/train_model.py --input_dir examen_dvc/data/processed --params examen_dvc/models/best_params.pkl --model_out examen_dvc/models/model.pkl
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def main(input_dir, params_path, model_out):
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))
    if y_train.shape[1] > 1:
        y_train = y_train.iloc[:, 0]

    params = joblib.load(params_path)
    model = RandomForestRegressor(random_state=42, **params)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print('Model saved to', model_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data/processed')
    parser.add_argument('--params', type=str, default='./models/best_params.pkl')
    parser.add_argument('--model_out', type=str, default='./models/model.pkl')
    args = parser.parse_args()
    main(args.input_dir, args.params, args.model_out)