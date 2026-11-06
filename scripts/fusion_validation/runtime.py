"""Portable data and output roots for supplementary fusion experiments."""
import os
from pathlib import Path
import re

REPOSITORY = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('LAIGN_WORKSPACE', REPOSITORY)).expanduser().resolve()
BASE = Path(os.environ.get('LAIGN_FUSION_OUTPUT', ROOT / 'outputs/fusion_validation')).expanduser().resolve()


def canonical_id(value):
    return re.sub(r'(?<=:)h\d{2}(?:_platinum)?__', '', str(value))


class PredictionArchive:
    """Normalize cohort identifiers without changing any numeric prediction."""
    def __init__(self, path, **kwargs):
        import numpy as np
        self._archive = np.load(path, **kwargs)
        self.files = self._archive.files

    def __getitem__(self, key):
        import numpy as np
        values = self._archive[key]
        if key.endswith('__sample_ids'):
            return np.asarray([canonical_id(v) for v in values], dtype=str)
        if key == 'sample_id':
            return np.asarray(canonical_id(values), dtype=str)
        return values

    def close(self):
        self._archive.close()

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def keys(self):
        return self.files

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_archive(path, **kwargs):
    return PredictionArchive(path, **kwargs)
