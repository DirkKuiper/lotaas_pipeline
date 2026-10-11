"""The ATNF catalogue, queried safely by many processes on one node.

psrqpy downloads the catalogue into the astropy cache in the user's home. On a
fresh CPU node the first batch started 64 classifiers at once; they downloaded
it together and one read a truncated copy ("Compressed file ended before the
end-of-stream marker was reached"), failing FETCH for the beam of J0323+3944.
The first query on a node now runs under a lock, and a query that fails clears
the cached copy and tries again.

The catalogue version is pinned: psrqpy would download whatever ATNF calls
"latest" onto a node without a cache, and nodes would disagree on which
pulsars are known. The first query on a node installs the pinned copy that
travels with the source whenever its cache holds another version or none.
"""
import fcntl
import os
from pathlib import Path
import shutil
import tarfile
import time

READY_SECONDS = 86400
CATALOGUE_VERSION = '2.8.1'
PINNED = Path(__file__).resolve().parent/'data'/f'psrcat_pkg-{CATALOGUE_VERSION}.tar.gz'


def _paths():
    home = Path(os.environ.get('HOME') or '/tmp')
    return home/'.lotaas-atnf.lock', home/'.lotaas-atnf.ready'


def catalogue_version(path):
    """The '#CATALOGUE' version of a psrcat_pkg tarball (None when unreadable)."""
    try:
        with tarfile.open(path) as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith('psrcat.db'))
            for line in archive.extractfile(member):
                if line.startswith(b'#CATALOGUE'):
                    return line.split()[1].decode()
                if not line.startswith(b'#'):
                    return None
    except (OSError, StopIteration, tarfile.TarError, IndexError):
        return None


def cache_dir():
    try:
        from psrqpy.utils import CACHEDIR
        return Path(CACHEDIR)
    except Exception:
        return Path(os.environ.get('HOME') or '/tmp')/'.cache'/'psrqpy'


def pin_catalogue(cache=None, pinned=PINNED):
    """Make psrqpy's cached catalogue the pinned version; True when it was replaced."""
    cache = Path(cache or cache_dir())
    target = cache/'psrcat_pkg.tar.gz'
    if not pinned.exists() or catalogue_version(target) == CATALOGUE_VERSION:
        return False
    cache.mkdir(parents=True, exist_ok=True)
    partial = cache/'psrcat_pkg.tar.gz.partial'
    shutil.copyfile(pinned, partial)
    os.replace(partial, target)
    (cache/'query_latest.pkl').unlink(missing_ok=True)   # parsed from the catalogue it replaced
    return True


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
                pin_catalogue()
                query = factory(**kwargs)
                ready.touch()
                return query
            except Exception as failure:
                error = failure
                _clear_cache()
        time.sleep(5 * (attempt + 1))
    raise error
