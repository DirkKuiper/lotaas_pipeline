"""Poll a persisted StageIT request and retrieve completed members incrementally."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from staging.client import StageIT, private_json, token_for_url
from euroflash.download import download, extract
from euroflash.ledger import Ledger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('request_directory',type=Path)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--poll-seconds',type=float,default=60)
    p.add_argument('--once',action='store_true')
    a=p.parse_args()
    state=json.loads((a.request_directory/'request.json').read_text())
    api=StageIT();ledger=Ledger(a.ledger)
    a.destination.mkdir(parents=True,exist_ok=True)
    def retrieve(url,manifest):
        target=a.destination/Path(urlsplit(url).path).name
        item=target.stem
        marker=target.with_suffix('.extracted.json')
        if ledger.completed(item,'retrieve','http-tar-v1'):
            return
        attempt=ledger.start(item,'retrieve','http-tar-v1',target.with_suffix('.receipt.json'),['StageIT',str(state['request_id']),url])
        try:
            receipt=download(url,token_for_url(manifest,url),target,64*2**30)
            fits=extract(target,a.destination/target.stem)
            marker.write_text(json.dumps({'request_id':state['request_id'],'archive':receipt,'fits':[str(p) for p in fits]},indent=2))
            matching=[u for u in state['surls'] if Path(urlsplit(u).path).name==target.name]
            if len(matching)!=1:
                raise ValueError('Ambiguous archive URL: cannot record retrieval provenance')
            ledger.record_archive(matching[0],state['request_id'],receipt,fits,marker)
            ledger.finish(attempt,[marker,*fits])
            print('Retrieved:',target.name,flush=True)
        except Exception as error:
            ledger.finish(attempt,error=str(error));print('Retrieve failed:',target.name,str(error),flush=True)
            raise
    while True:
        status=api.status(state['request_id'])
        private_json(a.request_directory/'status.json',status)
        response=json.loads(status['response'] or '{}')
        online={Path(urlsplit(u).path).name for u in response.get('online',[])}
        print('Request',state['request_id'],status['currentStatus'],len(online),'online',len(response.get('errors',{})),'errors',flush=True)
        if online:
            manifest=api.downloads(state['request_id'])
            private_json(a.request_directory/'downloads.json',manifest)
            urls=[u for u in manifest['urls'] if Path(urlsplit(u).path).name in online]
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                list(pool.map(lambda u:retrieve(u,manifest),urls))
        if status['currentStatus'].lower() in {'success','failed','aborted','partial success'}:
            if status['currentStatus'].lower()!='success':
                raise RuntimeError('Staging incomplete; inspect status.json')
            break
        if a.once:break
        time.sleep(a.poll_seconds)


if __name__=='__main__':
    main()
