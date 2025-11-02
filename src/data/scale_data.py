import argparse
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))

    # Supprime les colonnes non numériques (comme les dates)
    X_train = X_train.select_dtypes(include=['float64', 'int64'])
    X_test = X_test.select_dtypes(include=['float64', 'int64'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        os.path.join(output_dir, 'X_train_scaled.csv'), index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
        os.path.join(output_dir, 'X_test_scaled.csv'), index=False
    )

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(" Données normalisées sauvegardées avec succès.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./data/processed')
    parser.add_argument('--output_dir', type=str, default='./data/processed')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
