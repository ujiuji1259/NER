import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))
from utils import data_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT')
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--train_path', type=str, help='data path')
    parser.add_argument('--val_path', type=str, help='data path')
    args = parser.parse_args()
    train_data = data_utils.load_dataset(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size=0.05, random_state=42)

    train_output = '\n\n'.join(['\n'.join([t + '\t' + v for t,v in zip(x, y)]) for x, y in zip(X_train, y_train)])
    val_output = '\n\n'.join(['\n'.join([t + '\t' + v for t,v in zip(x, y)]) for x, y in zip(X_test, y_test)])

    with open(args.train_path, 'w') as f:
        f.write(train_output)
    with open(args.val_path, 'w') as f:
        f.write(val_output)

