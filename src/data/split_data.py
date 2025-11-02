"""
Split des données en X_train, X_test, y_train, y_test
Usage: python src/data/split_data.py --input_url <remote_csv_or_local> --output_dir examen_dvc/data/processed
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def main(input_path, output_dir, test_size=0.2, random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    # la cible est la dernière colonne
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./data/processed_data')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    main(args.input_path, args.output_dir, args.test_size, args.random_state)