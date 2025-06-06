import os

import yaml


def create_data_yaml(train_dir: str, val_dir: str, class_names: list[str], save_path: str):
    data = {
        'train': os.path.join('../../', train_dir),
        'val':   os.path.join('../../', val_dir),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(data, f)
    print(f"Data config YAML created at: {save_path}")