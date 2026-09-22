"""Audit valid, named PULP products against a recovered StageIT URL inventory.

Public DBView exposes catalogue metadata; it does not prove tape availability.
Cache each page so an interrupted audit can resume without repeating requests.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlencode, urlsplit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='LT5_004')
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    args=parser.parse_args();args.cache.mkdir(parents=True,exist_ok=True)
    params={'mode':'table_res','class_str':'PulpDataProduct','project':args.project,
            'Exportselect':'CSV','mainpref_numrows':10000,'QSort':'filename','QDesc':'ascending',
            'PulpDataProduct.isValid.op':'=','PulpDataProduct.isValid':1}
    records=[];previous='';page=1
    while True:
        path=args.cache/f'valid-pulp-{page:04d}.csv'
        if not path.exists():
            partial=path.with_suffix('.partial')
            url='https://lta-dbview.lofar.eu/DbView?'+urlencode(params)
            subprocess.run(['wget','-q','-4','--timeout=120','--tries=2',url,'-O',str(partial)],check=True)
            if not partial.read_text().startswith('ROWNUM,object_id,'):
                raise ValueError('DBView did not return a catalogue CSV')
            partial.replace(path)
        rows=list(csv.DictReader(path.open()))
        if any(row['isValid']!='1' for row in rows):
            raise ValueError('Catalogue validity filter was not applied')
        # NULL names cannot be staged or paginated by filename. Report them
        # separately rather than treating the query as a complete LTA inventory.
        named=[row for row in rows if row['filename']]
        if named and (named[0]['filename']<=previous or
                      [r['filename'] for r in named] != sorted(r['filename'] for r in named)):
            raise ValueError('Nonunique/nonmonotonic filename pagination; use a catalogue export')
        if len(rows)==10000 and len(named)==len(rows) and sum(r['filename']==named[-1]['filename'] for r in named)>1:
            raise ValueError('Duplicate filename at a page boundary; a catalogue export is required')
        records.extend(named)
        print('Catalogue named valid products:',len(records),flush=True)
        if len(rows)<10000 or len(named)<len(rows):break
        previous=named[-1]['filename']
        params.update({'PulpDataProduct.filename.op':'>','PulpDataProduct.filename':previous})
        page+=1
    archive_names={re.sub(r'_([0-9a-fA-F]{8})\.tar$', '.tar',Path(urlsplit(u).path).name)
                   for u in args.inventory.read_text().splitlines() if u.strip()}
    named={re.sub(r'_([0-9a-fA-F]{8})\.tar$', '.tar',r['filename']) for r in records}
    missing=sorted(named-archive_names)
    result={'project':args.project,'valid_named_products':len(records),
            'unique_logical_filenames':len(named),
            'known_archive_matches':len(named & archive_names),'missing_archive_locations':len(missing),
            'catalogue_complete':False,'scope':'Valid named PULP metadata only; excludes other product types and unnamed records. File availability requires StageIT.',
            'missing_filenames':missing}
    (args.cache/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='missing_filenames'},indent=2))


if __name__=='__main__':
    main()
