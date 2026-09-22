"""Recover known project URLs from StageIT history (not a full LTA catalogue)."""
import argparse
import json
from pathlib import Path
import re
from staging.client import StageIT
from euroflash.ledger import Ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', default='lt5_004')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--max-pages', type=int, default=20)
    a = p.parse_args()
    api = StageIT()
    cursor = None
    urls = set()
    complete = False
    for page in range(a.max_pages):
        after = ', after: '+json.dumps(cursor) if cursor else ''
        data = api.query('{ requests(first: 100%s) { pageInfo { hasNextPage endCursor } edges { node { surls { surl } } } } }' % after)['requests']
        for edge in data['edges']:
            for obj in edge['node']['surls']:
                url = obj['surl']
                if f'/{a.project.lower()}/' in url.lower() and re.search(r'_bf_[0-9a-f]{8}\.tar$', url):
                    urls.add(url)
        print(f'History page {page+1}: {len(urls)} known {a.project} files', flush=True)
        if not data['pageInfo']['hasNextPage']:
            complete = True
            break
        new_cursor = data['pageInfo']['endCursor']
        if new_cursor == cursor:
            raise RuntimeError('StageIT pagination did not advance')
        cursor = new_cursor
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(sorted(urls))+'\n')
    metadata = {'project': a.project, 'files': len(urls), 'source': 'StageIT request history',
                'history_complete': complete, 'catalogue_complete': False,
                'availability': 'unknown until checked/staged'}
    a.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    Ledger(a.ledger).discover(urls, a.project, metadata['source'])


if __name__ == '__main__':
    main()
