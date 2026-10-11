"""Round 1: validate NSL-KDD and prepare leakage-free search inputs."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
]
CATEGORICAL = ['protocol_type', 'service', 'flag']
NUMERIC = [name for name in FEATURE_NAMES if name not in CATEGORICAL]


def read_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path.name}. Put it in {ROOT / "data"}.')
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != 43:
        raise ValueError(f'{path.name}: expected 43 columns, found {frame.shape[1]}.')
    frame.columns = FEATURE_NAMES + ['label', 'difficulty']
    if frame.isna().any().any():
        raise ValueError(f'{path.name}: missing values found; inspect the download.')
    for name in CATEGORICAL:
        frame[name] = frame[name].astype(str).str.strip()
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    if not np.isfinite(frame[NUMERIC].to_numpy()).all():
        raise ValueError(f'{path.name}: non-finite numeric values.')
    labels = frame['label'].astype(str).str.strip().str.rstrip('.')
    targets = (labels != 'normal').astype(np.int8).to_numpy()
    return frame[FEATURE_NAMES].copy(), targets


def transform_features(frame, encoder):
    numeric = frame[NUMERIC].to_numpy(dtype=np.float32)
    categorical = encoder.transform(frame[CATEGORICAL]).astype(np.float32)
    return np.concatenate([numeric, categorical], axis=1)


def mask_to_columns(mask, feature_groups):
    mask = np.asarray(mask)
    if mask.shape != (41,) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Mask must contain exactly 41 binary entries.')
    if not mask.any():
        raise ValueError('Select at least one feature.')
    return [column for bit, name in zip(mask, FEATURE_NAMES)
            if bit for column in feature_groups[name]]


def class_counts(targets):
    return {'normal': int((targets == 0).sum()), 'attack': int((targets == 1).sum())}


def prepare_data(root=ROOT):
    root = Path(root)
    raw_train, y_full = read_dataset(root / 'data' / 'KDDTrain+.txt')
    raw_test, y_test = read_dataset(root / 'data' / 'KDDTest+.txt')
    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        raw_train, y_full, test_size=0.2, random_state=42, stratify=y_full)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                            dtype=np.float32)
    encoder.fit(x_train_raw[CATEGORICAL])
    groups = {name: [index] for index, name in enumerate(NUMERIC)}
    offset = len(NUMERIC)
    for name, categories in zip(CATEGORICAL, encoder.categories_):
        groups[name] = list(range(offset, offset + len(categories)))
        offset += len(categories)
    x_train = transform_features(x_train_raw, encoder)
    x_val = transform_features(x_val_raw, encoder)
    x_test = transform_features(raw_test, encoder)
    indices = np.arange(len(y_train))
    if len(indices) > 20_000:
        indices, _ = train_test_split(indices, train_size=20_000,
                                     random_state=42, stratify=y_train)
    bundle = {
        'X_train': x_train, 'y_train': y_train,
        'X_search': x_train[indices], 'y_search': y_train[indices],
        'X_val': x_val, 'y_val': y_val, 'X_test': x_test, 'y_test': y_test,
        'raw_train': raw_train, 'raw_test': raw_test, 'y_full': y_full,
        'encoder': encoder, 'feature_names': FEATURE_NAMES,
        'numeric_features': NUMERIC, 'categorical_features': CATEGORICAL,
        'feature_groups': groups, 'split_seed': 42,
        'train_attack_types': pd.read_csv(root / 'data' / 'KDDTrain+.txt', header=None,
                                          usecols=[41]).iloc[:, 0].astype(str).str.strip().str.rstrip('.').to_numpy(),
        'test_attack_types': pd.read_csv(root / 'data' / 'KDDTest+.txt', header=None,
                                         usecols=[41]).iloc[:, 0].astype(str).str.strip().str.rstrip('.').to_numpy(),
    }
    summary = {
        'dataset': 'NSL-KDD', 'original_features': 41,
        'encoded_features': int(x_train.shape[1]),
        'source_training_rows': len(raw_train), 'training_rows': len(y_train),
        'search_training_rows': len(indices), 'validation_rows': len(y_val),
        'test_rows': len(y_test), 'feature_names': FEATURE_NAMES,
        'class_counts': {'train': class_counts(y_train),
                         'validation': class_counts(y_val), 'test': class_counts(y_test)},
        'expected_full_files': len(raw_train) == 125_973 and len(raw_test) == 22_544,
    }
    output = root / 'results'
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / 'prepared_data.joblib', compress=3)
    (output / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    try:
        summary = prepare_data()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))
    print(json.dumps(summary, indent=2))
    print('\nPrepared data and dataset summary saved in results/.')
    if not summary['expected_full_files']:
        print('WARNING: row counts differ from the standard full NSL-KDD files.')
