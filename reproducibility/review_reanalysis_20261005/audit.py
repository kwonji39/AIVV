"""Offline reanalysis only; no new training or hosted-model calls.
Usage: python audit.py [path/to/Supplementary_Software_1]
"""
import json,gzip,hashlib,csv,math,statistics,sys
from pathlib import Path
from collections import defaultdict,Counter
W=Path(__file__).resolve().parent
R=Path(sys.argv[1]) if len(sys.argv)>1 else W.parent/'Supplementary_Software_1'
U=json.loads((R/'uuv_traces.json').read_text()); O=W/'analysis';O.mkdir(exist_ok=True)
def write(name,rows):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def exact(b,c):
 n=b+c
 return min(1,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1
rows=[];checks=[];prom=[];votes=Counter();pairs=Counter();nround=0;examples=[];round2=Counter()
for r in U:
 f=list(map(bool,r['fail']));lab=list(map(bool,r['labels']));a=lab.index(True);b=len(lab)-1-lab[::-1].index(True);persistent=r['dataset']=='DS3'
 assert len(f)==410 and (a,b)==(209,259)
 truth=[(i>=a if persistent else lab[i]) for i in range(len(f))]
 nuisance=sum(f[:a])+ (0 if persistent else sum(f[b+1:]));hit=any(f[a:]);short_hit=any(f[a:b+1]);success=bool(hit and nuisance==0)
 raw=gzip.decompress((R/r['source']).read_bytes());assert hashlib.sha256(raw).hexdigest()==r['sha256'];t=json.loads(raw)
 rows.append(dict(dataset=r['dataset'],stage=r['stage'],seed=r['seed'],n=len(f),event_success=int(success),injection_window_hit=int(short_hit),literal_terminal_success=int(success and not f[-1]),terminal_fail=int(f[-1]),nuisance=nuisance,lead_fail_per_1000=1000*sum(f[:a])/a,post_fail_fraction=sum(f[b+1:])/len(f[b+1:]),fault_coverage=sum(x and y for x,y in zip(f,truth))/sum(truth),sample_agreement=sum(x==y for x,y in zip(f,truth))/len(f),source=r['source'],sha256=r['sha256']))
 if r['stage'].startswith('baseline'):
  checks.append(dict(stage=r['stage'],seed=r['seed'],initial_flags=sum(t['initial_fail']),final_flags=sum(t['final_fail']),overrides=sum(x and not y for x,y in zip(t['initial_fail'],t['final_fail'])),new_fail=sum(not x and y for x,y in zip(t['initial_fail'],t['final_fail'])),raw_council_flags=sum(t['raw_council_fail'])))
 if r['stage']!='full':continue
 n_prom=0;fault_prom=0;attempt=0;prom_samples=[]
 for i,s in enumerate(t['sample_logs']):
  ds=s['agent_decisions'];ss=[d for d in ds if d['agent_name']=='sentry'];init=ss[0]['decision']=='FAIL'
  attempt+=any(d['agent_name']=='tuner' for d in ds)
  promoted=len(ss)>1 and ss[-1]['decision']=='PASS';n_prom+=promoted;fault_prom+=promoted and truth[i]
  if promoted:prom_samples.append(i)
  groups=[];cur={}
  for d in ds:
   if d['agent_name'] in ['req_eng','fail_mgr','sys_eng']:
    cur[d['agent_name']]=d['decision']=='FAIL'
    if len(cur)==3:
     groups.append(cur);cur={}
  if len(groups)>1:
   round2[(r['dataset'],'attempts')]+=1
   round2[(r['dataset'],'final_pass')]+=s['final_decision']=='PASS'
  if r['dataset']=='DS1':
   for g in groups:
    nround+=1
    for k,v in g.items():votes[k]+=v!=truth[i]
    for p in [('req_eng','fail_mgr'),('req_eng','sys_eng'),('fail_mgr','sys_eng')]:pairs['/'.join(p)]+=g[p[0]]==g[p[1]]
  if r['dataset']=='DS2' and r['seed']==0 and i==212:examples.append(s)
 prom.append(dict(dataset=r['dataset'],seed=r['seed'],attempts=attempt,promotions=n_prom,fault_promotions=fault_prom,promotion_sample_ids=';'.join(map(str,prom_samples))))
write('uuv_per_run.csv',rows);write('baseline_overrides.csv',checks);write('adaptation_per_run.csv',prom)
summ=[]
for k in sorted(set((r['dataset'],r['stage']) for r in rows)):
 rr=[r for r in rows if (r['dataset'],r['stage'])==k];out=dict(dataset=k[0],stage=k[1],n=len(rr))
 for key in ['event_success','literal_terminal_success','injection_window_hit','terminal_fail']:out[key]=sum(r[key] for r in rr)
 for key in ['nuisance','lead_fail_per_1000','post_fail_fraction','fault_coverage','sample_agreement']:
  vals=[r[key] for r in rr];out[key+'_mean']=statistics.mean(vals);out[key+'_sd']=statistics.stdev(vals)
 summ.append(out)
write('uuv_summary.csv',summ)
bases=sorted(set(r['stage'] for r in rows if r['stage'].startswith('baseline')));tests=[];full={r['seed']:r for r in rows if r['dataset']=='DS1' and r['stage']=='full'}
for stage in bases:
 rr=[r for r in rows if r['stage']==stage];b=sum(full[r['seed']]['event_success']>r['event_success'] for r in rr);c=sum(full[r['seed']]['event_success']<r['event_success'] for r in rr);tests.append(dict(comparison='full vs '+stage,n=len(rr),full_only_success=b,baseline_only_success=c,p=exact(b,c)))
order=sorted(range(len(tests)),key=lambda j:tests[j]['p']);v=0
for rank,j in enumerate(order):v=max(v,min(1,(len(tests)-rank)*tests[j]['p']));tests[j]['p_holm']=v
write('paired_tests.csv',tests)
summary={'groups':summ,'baseline_checks':{s:{k:sum(x[k] for x in checks if x['stage']==s) for k in ['initial_flags','final_flags','overrides','new_fail']} for s in bases},'promotions':{},'round2':{'/'.join(k):v for k,v in round2.items()},'ds1_vote_rounds':nround,'vote_errors':dict(votes),'pair_agreement':dict(pairs),'tests':tests}
for d in ['DS1','DS2','DS3']:
 rr=[r for r in prom if r['dataset']==d];summary['promotions'][d]={k:sum(r[k] for r in rr) for k in ['attempts','promotions','fault_promotions']}
(O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'example_DS2_seed0_sample212.json').write_text(json.dumps(examples,indent=2))
print(json.dumps(summary,indent=2))
# Complete council diagnostics: all rounds in the audited DS1 full-pipeline cohort.
joint=Counter();majority_errors=0;fallback_mentions=Counter();unique_escalations=Counter()
for r in U:
 if r['stage']!='full':continue
 t=json.loads(gzip.decompress((R/r['source']).read_bytes()))
 for s in t['sample_logs']:
  ds=s['agent_decisions'];unique_escalations[r['dataset']]+=any(d['agent_name']=='req_eng' for d in ds)
  for d in ds:
   if any(k in d['reasoning'].lower() for k in ['fallback','parse error','api error','error occurred']):fallback_mentions[r['dataset']+'/'+d['agent_name']]+=1
  if r['dataset']!='DS1':continue
  g={}
  for d in ds:
   if d['agent_name'] in ['req_eng','fail_mgr','sys_eng']:
    g[d['agent_name']]=d['decision']=='FAIL'
    if len(g)==3:
     truth=bool(s['actual_label']);key=''.join(str(int(g[k]!=truth)) for k in ['req_eng','fail_mgr','sys_eng']);joint[key]+=1;majority_errors+=(sum(g.values())>=2)!=truth;g={}
summary.update(joint_errors=dict(joint),majority_errors=majority_errors,unique_escalations=dict(unique_escalations),fallback_text_mentions=dict(fallback_mentions))
# Post-hoc persistence sensitivity changes only the score, not execution or prompts.
S=json.loads((R/'satellite_traces.json').read_text());sens=[]
def regions(f,p):
 out=[];i=0
 while i<len(f):
  if not f[i]:i+=1;continue
  j=i+1
  while j<len(f) and f[j]:j+=1
  if j-i>=p:out.append((i,j-1))
  i=j
 return out
for ch in ['E-8','F-5','T-4','D-1']:
 rr=[r for r in S if r['channel']==ch]
 for p in sorted(set([1,2,4,5,10,rr[0]['persistence']])):
  count=0;no_release=0
  for r in rr:
   a,b=[v-r['global_start'] for v in r['window']];reg=regions(r['fail'],p);hit=any(u<=b and v>=a for u,v in reg);lead=any(v<a for u,v in reg);ok=hit and not lead;no_release+=ok;count+=ok and (b>=len(r['fail'])-1 or not r['fail'][-1])
  sens.append(dict(channel=ch,score_persistence=p,execution_persistence=rr[0]['persistence'],n=len(rr),fvr=count,without_terminal_release=no_release))
write('spacecraft_score_sensitivity.csv',sens)
(O/'summary.json').write_text(json.dumps(summary,indent=2))
print('Additional diagnostics',summary['joint_errors'],'majority errors',majority_errors,'escalations',dict(unique_escalations))
