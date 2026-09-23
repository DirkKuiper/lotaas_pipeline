"""The ATNF catalogue, queried safely by many processes on one node.

psrqpy downloads the catalogue into the astropy cache in the user's home. On a
fresh CPU node the first batch started 64 classifiers at once; they downloaded
it together and one read a truncated copy ("Compressed file ended before the
end-of-stream marker was reached"), failing FETCH for the beam of J0323+3944.
The first query on a node now runs under a lock, and a query that fails clears
the cached copy and tries again.
"""
import fcntl
import os
from pathlib import Path
import time

READY_SECONDS = 86400


def _paths():
    home = Path(os.environ.get('HOME') or '/tmp')
    return home/'.lotaas-atnf.lock', home/'.lotaas-atnf.ready'


def _clear_cache():
    try:
        from astropy.utils.data import clear_download_cache, get_cached_urls
        for url in get_cached_urls():
            if 'atnf' in url or 'psrcat' in url:
                clear_download_cache(url)
    except Exception:
        pass


def query_atnf(attempts=3, factory=None, **kwargs):
    """psrqpy.QueryATNF(**kwargs), serialised until the node has a good cached catalogue."""
    if factory is None:
        from psrqpy import QueryATNF as factory
    lock, ready = _paths()
    if ready.exists() and time.time() - ready.stat().st_mtime < READY_SECONDS:
        try:
            return factory(**kwargs)
        except Exception:
            pass   # take the locked path below, which repairs the cache
    error = None
    for attempt in range(attempts):
        with lock.open('w') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                query = factory(**kwargs)
                ready.touch()
                return query
            except Exception as failure:
                error = failure
                _clear_cache()
        time.sleep(5 * (attempt + 1))
    raise error
