"""Run the web layer: `python -m web serve|index|snippets|url` (see web/README.md)."""
import argparse
import json
import logging
import sys

from web import config


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m web', description=__doc__)
    parser.add_argument('--config', help='TOML overrides (default ~/.config/lotaas/web.toml)')
    sub = parser.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve', help='Serve the pages, keeping the index and snippets current')
    serve.add_argument('--host')
    serve.add_argument('--port', type=int)
    serve.add_argument('--no-background', action='store_true', help='Serve only; do not index or cut')
    index = sub.add_parser('index', help='One index pass (or --watch)')
    index.add_argument('--full', action='store_true', help='Rescan every results directory')
    index.add_argument('--watch', action='store_true')
    snippets = sub.add_parser('snippets', help='One hold/cut/release pass')
    snippets.add_argument('--rescan', action='store_true', help='Look for source filterbanks again')
    sub.add_parser('url', help='Print the address with its access token')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    cfg = config.load(args.config, **({'host': args.host, 'port': args.port} if args.command == 'serve' else {}))

    if args.command == 'serve':
        import uvicorn
        from web.app import create_app
        from web.app import token
        app = create_app(cfg, run_background=not args.no_background)
        print(f'Open http://localhost:{cfg.port}/?token={token(cfg)}', flush=True)
        uvicorn.run(app, host=cfg.host, port=cfg.port, log_level='warning')
    elif args.command == 'index':
        import time
        from web.indexer import Indexer
        indexer = Indexer(cfg)
        while True:
            print(json.dumps(indexer.run_pass(full=args.full)), flush=True)
            if not args.watch:
                break
            time.sleep(cfg.index_seconds)
    elif args.command == 'snippets':
        from web.indexer import Indexer
        from web.snippets import Snippets
        indexer = Indexer(cfg)
        indexer.run_pass()
        cutter = Snippets(cfg)
        if args.rescan:
            cutter.sources_scanned = 0
        print(json.dumps(cutter.run_pass()), flush=True)
        indexer.sync_snippets()
        indexer.derive()
    elif args.command == 'url':
        from web.app import token
        print(f'http://localhost:{cfg.port}/?token={token(cfg)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
