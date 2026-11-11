import csv
import json
from pathlib import Path

import numpy as np


def save_history(frame, variable_ids, path=None, *, append=True):
    """Write one iteration; JSON arrays preserve mean and covariance shapes."""
    path = (Path.cwd() / 'history.csv' if path is None else Path(path)).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0

    # Closing after each iteration preserves completed records on solver errors.
    encoding = 'utf-8-sig' if write_header else 'utf-8'
    with path.open('a' if append else 'w', newline='', encoding=encoding) as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['iteration', 'variable_id', 'mean', 'cov'])
        if frame is not None:
            for variable_id, mean, cov in zip(variable_ids, frame['means'], frame['covs']):
                writer.writerow([
                    frame['iter'], variable_id,
                    json.dumps(np.asarray(mean).tolist(), separators=(',', ':')),
                    json.dumps(np.asarray(cov).tolist(), separators=(',', ':')),
                ])
    return path
