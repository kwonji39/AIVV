"""Offline Experiment 2 reanalysis for the common seeds 0–14.
All original files are hash-checked; Experiment 1 is preserved, not reanalyzed.
"""
from pathlib import Path
import json,gzip,hashlib,csv,statistics
R=Path(__file__).resolve().parent;O=R/'source_data';O.mkdir(exist_ok=True)
S=json.loads((R/'satellite_traces.json').read_text());U=json.loads((R/'uuv_traces.json').read_text())
def read(r):
 raw=gzip.decompress((R/r['source']).read_bytes())
 assert hashlib.sha256(raw).hexdigest()==r['sha256'],r['source']
 return json.loads(raw)
def regions(flags,persistence):
 regions=[];i=0
 while i<len(flags):
  if not flags[i]:i+=1;continue
  j=i+1
  while j<len(flags) and flags[j]:j+=1
  if j-i>=persistence:regions.append((i,j-1))
  i=j
 return regions
def write(name,rows):
 with (O/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for r in U:read(r)
rows=[]
for r in S:
 d=read(r);logs=d['sample_logs'];flags=[x['final_decision']=='FAIL' for x in logs]
 assert flags==r['fail'] and r['complete']
 assert len(logs)==r['n_total']-r['global_start']
 a,b=r['window'];offset=r['global_start'];rr=[(x+offset,y+offset) for x,y in regions(flags,r['persistence'])]
 success=any(x<=b and y>=a for x,y in rr) and not any(y<a for x,y in rr) and (b==r['n_total']-1 or not flags[-1])
 assert success==r['success']
 council=sum(any(x['agent_name']=='req_eng' for x in l['agent_decisions']) for l in logs)
 adaptation=sum(any(x['agent_name']=='inspector' for x in l['agent_decisions']) for l in logs)
 assert {'req_eng':council,'inspector':adaptation}==r['counts']
 rows.append(dict(channel=r['channel'],seed=r['seed'],test_samples=len(flags),success=int(success),council_samples=council,council_share=100*council/len(flags),adaptation_invocations=adaptation))
rows.sort(key=lambda r:(['E-8','F-5','T-4','D-1'].index(r['channel']),r['seed']))
write('experiment2_per_run',rows)
summary=[]
for channel in ['E-8','F-5','T-4','D-1']:
 sub=[r for r in rows if r['channel']==channel]
 assert [r['seed'] for r in sub]==list(range(15))
 wins=sum(r['success'] for r in sub);assert wins==(12 if channel=='E-8' else 15)
 summary.append(dict(channel=channel,seeds='0-14',n=15,successes=wins,fvr_percent=100*wins/15,test_samples=sub[0]['test_samples'],council_samples_mean=statistics.mean(r['council_samples'] for r in sub),council_share_percent=statistics.mean(r['council_share'] for r in sub),adaptation_invocations_mean=statistics.mean(r['adaptation_invocations'] for r in sub)))
assert [r['seed'] for r in rows if r['channel']=='E-8' and not r['success']]==[8,10,14]
write('table4_and_supp_table3',summary)
checks={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((R/'data').rglob('*')) if p.is_file()}
expected=json.loads((O/'input_checksums.json').read_text())
for path,digest in checks.items():
 assert expected[path]==digest,path
(O/'input_checksums_recomputed.json').write_text(json.dumps(checks,indent=2))
verification=dict(status='PASS',original_file_hashes_verified=len(U)+len(S),experiment2_complete_runs=len(S),experiment2_seeds=list(range(15)),experiment2_results=summary,experiment1='Original author-reported results retained; no new metric or independent reproduction claim.')
(O/'verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification,indent=2))
