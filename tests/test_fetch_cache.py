from concurrent.futures import ThreadPoolExecutor
import threading
import time

from lotaas_reprocessing.fetch_models import load_models


def test_cold_workers_do_not_download_the_same_weight_file_concurrently(tmp_path):
    guard = threading.Lock()
    writers, downloads = set(), []
    def factory(name):
        path = tmp_path/(name + '.weights')
        if not path.exists():
            with guard:
                assert name not in writers, 'a second writer would corrupt the shared model file'
                writers.add(name)
            time.sleep(.03)
            path.write_text('validated weights')
            with guard:
                writers.remove(name)
                downloads.append(name)
        return path.read_text()
    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(lambda _: load_models(['a', 'b'], factory, tmp_path), range(8)))
    assert downloads == ['a', 'b']
    assert models == [{'a': 'validated weights', 'b': 'validated weights'}] * 8


def test_a_corrupted_warm_cache_is_repaired_under_the_lock(tmp_path):
    calls = []
    def factory(name):
        calls.append(name)
        if len(calls) == 2:
            raise ValueError('corrupt weight checksum')
        return 'model'
    assert load_models(['a'], factory, tmp_path) == {'a': 'model'}
    assert load_models(['a'], factory, tmp_path) == {'a': 'model'}
    assert calls == ['a', 'a', 'a']
