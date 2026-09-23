"""Publish completed periodicity folds from the head node, explicitly and once.

Pilot products require --include-pilot and are labelled as validation. Catalogue
associations are reported as associations; neither the FFT statistic nor a Slack
message is treated as a validated new pulsar discovery.
"""
import argparse
import hashlib
import json
from pathlib import Path
from lotaas_reprocessing.trials import product_outputs
from postproc.notify_candidates import connect,record_sent,sent_keys
from postproc.slack_client import Slack


def discover(roots,include_pilot=False,known_only=False):
    for root in roots:
        root=Path(root)
        summaries=[root/'periodicity_summary.json'] if (root/'periodicity_summary.json').is_file() else sorted(root.rglob('periodicity_summary.json'))
        for summary in summaries:
            directory=summary.parent
            product_outputs(directory,summary.name)
            metadata=json.loads((directory/'metadata.json').read_text())
            if metadata.get('pilot',True) and not include_pilot:continue
            item=Path(metadata['filename']).stem
            for line in (directory/'periodicity_folded_candidates.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['rfi_like']:continue
                if known_only and not row.get('catalogue_matches'):continue
                row.update(path=directory/row['plot'],item=item,pilot=metadata.get('pilot',True),
                           fingerprint=directory.name)
                row['key']='periodicity|'+hashlib.sha256((item+'|'+directory.name+'|'+row['plot']).encode()).hexdigest()
                yield row


def message(row):
    names=', '.join(m['name'] for m in row.get('catalogue_matches',[]))
    title='*Periodicity validation candidate*' if row['pilot'] else '*Periodic candidate*'
    lines=[title,row['item'],f"Period = {row['refined_period_seconds']:.9f} s (topocentric)",
           f"DM = {row['dm']:.3f} pc/cm³",
           f"FFT −log10(p) = {row['statistic']:.2f} (nominal; not Gaussian S/N)",
           f"Harmonics = {row['harmonic_count']}; valid duration = {row['observation_seconds']:.1f} s",
           f"Catalogue association: {names or 'none'}"]
    if row['pilot']:lines.append('_Pilot validation; not production survey coverage._')
    lines.append('Zero-acceleration search; folded profile, time persistence and period refinement shown.')
    return '\n'.join(lines)


def run_once(slack,db,roots,limit=1,dry_run=False,include_pilot=False,known_only=False):
    already=sent_keys(db);posted=[]
    for row in discover(roots,include_pilot,known_only):
        if row['key'] in already:continue
        if len(posted)>=limit:break
        text=message(row)
        if dry_run:
            print(text);print(row['path']);file_id=None
        else:
            # Require a recorded successful periodicity stage for this beam/run.
            success=db.execute("SELECT 1 FROM attempts WHERE item=? AND stage='periodicity' "
                               "AND status='success' AND fingerprint LIKE ?",
                               (row['item'],row['fingerprint']+'%')).fetchone()
            if not success:raise ValueError('No successful periodicity ledger attempt for this plot')
            file_id=slack.upload(row['path'],title='Periodic search — '+row['item'],comment=text)
            if not file_id:raise RuntimeError('Slack did not confirm the upload')
            record_sent(db,row['key'],'periodicity',row['item'],str(row['path']),file_id,slack.channel)
        posted.append({'plot':str(row['path']),'file_id':file_id,'key':row['key']})
        already.add(row['key'])
    return posted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots',type=Path,nargs='+')
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--channel')
    parser.add_argument('--include-pilot',action='store_true')
    parser.add_argument('--known-only',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--limit',type=int,default=1)
    args=parser.parse_args()
    if args.limit<1:parser.error('--limit must be positive')
    slack=Slack(channel=args.channel)
    if not args.dry_run:
        if not slack.enabled:parser.error('Slack credentials or destination missing')
        print(json.dumps(slack.check()))
    with connect(args.ledger) as db:
        print(json.dumps(run_once(slack,db,args.roots,args.limit,args.dry_run,args.include_pilot,args.known_only),indent=2))


if __name__=='__main__':main()
