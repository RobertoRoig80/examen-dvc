"""
GridSearch pour trouver les meilleurs hyperparamètres d'un modèle de régression.
Lecture des paramètres depuis params.yaml.
Usage:
python src/models/grid_search.py --input_dir examen_dvc/data/processed --model_out examen_dvc/models/best_params.pkl --params_file params.yaml
"""

import argparse
import os
import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def main(input_dir, model_out, params_file):
    # Charger les données
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))

    # Si y_train est un DataFrame à une seule colonne
    if y_train.shape[1] > 1:
        y_train = y_train.iloc[:, 0]

    # Charger les paramètres depuis params.yaml
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)

    # Lire la section gridsearch
    grid_params = params.get("gridsearch", {})
    model_name = grid_params.get("model", "RandomForestRegressor")
    param_grid = grid_params.get("param_grid", {})

    # Sélectionner le modèle
    if model_name == "RandomForestRegressor":
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError(f"Modèle non supporté: {model_name}")

    # Lancer le GridSearch
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    # Sauvegarder les meilleurs paramètres
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(grid.best_params_, model_out)

    print('Best params:', grid.best_params_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data/processed')
    parser.add_argument('--model_out', type=str, default='./models/best_params.pkl')
    parser.add_argument('--params_file', type=str, default='params.yaml')
    args = parser.parse_args()

    main(args.input_dir, args.model_out, args.params_file)
