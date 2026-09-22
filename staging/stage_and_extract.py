#!/usr/bin/env python3
"""Submit once, persist the request ID, and resume polling without restaging."""
import argparse
import json
from pathlib import Path
import time
try:
    from .client import StageIT, private_json
except ImportError:
    from client import StageIT, private_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('srm_list', type=Path)
    p.add_argument('output_directory', type=Path)
    p.add_argument('--config')
    p.add_argument('--poll-seconds', type=float, default=60)
    p.add_argument('--timeout-hours', type=float, default=24)
    p.add_argument('--submit-only', action='store_true')
    a = p.parse_args()
    api = StageIT(a.config)
    surls = list(dict.fromkeys(line.strip() for line in a.srm_list.read_text().splitlines()
                               if line.strip() and not line.lstrip().startswith('#')))
    a.output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = a.output_directory / 'request.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['surls'] != surls:
            raise ValueError('Existing request has different input; use a new staging directory')
    else:
        state = {'request_id': api.submit(surls), 'surls': surls}
        private_json(state_path, state)
    print('StageIT request:', state['request_id'], flush=True)
    if a.submit_only:
        return
    deadline = time.monotonic() + a.timeout_hours * 3600
    while True:
        result = api.status(state['request_id'])
        private_json(a.output_directory / 'status.json', result)
        status = result['currentStatus'].lower()
        print('StageIT status:', status, flush=True)
        if status == 'success':
            manifest = api.downloads(state['request_id'])
            if len(manifest['urls']) != len(surls):
                raise RuntimeError('StageIT returned fewer URLs than requested')
            private_json(a.output_directory / 'downloads.json', manifest)
            (a.output_directory / 'webdav_links.txt').write_text('\n'.join(manifest['urls']) + '\n')
            if len(manifest['macaroons']) == 1:
                token_path = a.output_directory / 'macaroon.txt'
                token_path.touch(mode=0o600)
                token_path.chmod(0o600)
                token_path.write_text(manifest['macaroons'][0]['content'])
            return
        if status in {'failed', 'aborted', 'partial success'}:
            raise RuntimeError(f'Staging {status}; inspect status.json. Incomplete data will not be processed silently.')
        if time.monotonic() >= deadline:
            raise TimeoutError('Staging deadline reached; rerun to resume the same request')
        time.sleep(a.poll_seconds)


if __name__ == '__main__':
    main()
