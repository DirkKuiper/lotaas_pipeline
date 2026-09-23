"""Serialize first-use FETCH weight downloads on each compute node."""
import fcntl
import hashlib
from pathlib import Path
import time


def load_models(names, factory=None, cache_root=None):
    if factory is None:
        from fetch.utils import get_model as factory
    root = Path(cache_root) if cache_root else Path.home()
    signature = hashlib.sha256(','.join(sorted(names)).encode()).hexdigest()[:12]
    ready = root/('.lotaas-fetch-ready-' + signature)
    lock = root/'.lotaas-fetch.lock'
    if ready.exists():
        try:
            return {name: factory(name) for name in names}
        except Exception:
            # Keras validates cached hashes itself. Repair under the lock if
            # a prior interrupted download left a bad file behind.
            ready.unlink(missing_ok=True)
    error = None
    for attempt in range(3):
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if ready.exists():
                # Another worker populated the cache while this one waited.
                # Release the lock before constructing models from valid files.
                break
            try:
                models = {name: factory(name) for name in names}
                ready.touch()
                return models
            except Exception as failure:
                error = failure
        time.sleep(attempt + 1)
    else:
        raise error
    return {name: factory(name) for name in names}
