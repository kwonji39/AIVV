"""Second pass: reconstruct reported quantities directly from raw records.
No imports from audit.py; explicit independent score calculations.
"""
from pathlib import Path
import json,gzip,csv,math,statistics,sys
W=Path(__file__).resolve().parent;R=Path(sys.argv[1]) if len(sys.argv)>1 else W.parent/'Supplementary_Software_1';records=json.loads((R/'uuv_traces.json').read_text());summary=json.loads((W/'analysis/summary.json').read_text());checks=[]
for ds in ['DS1','DS2','DS3']:
 for stage in ['sentry','full']:
  success=literal=terminal=prom=faultprom=0;agreements=[];post=[]
  for rec in records:
   if (rec['dataset'],rec['stage'])!=(ds,stage):continue
   raw=json.load(gzip.open(R/rec['source'],'rt'));samples=raw['sample_logs'];f=[q['final_decision']=='FAIL' for q in samples];truth=[209<=i<=(409 if ds=='DS3' else 259) for i in range(410)]
   assert f==list(map(bool,rec['fail']))
   detected=any(f[209:260]);quiet=not any(f[:209]) and (ds=='DS3' or not any(f[260:]));ok=detected and quiet
   success+=ok;literal+=ok and not f[-1];terminal+=f[-1];agreements.append(sum(x==y for x,y in zip(f,truth))/410);post.append(sum(f[260:])/150)
   if stage=='full':
    for i,s in enumerate(samples):
     tun=[d for d in s['agent_decisions'] if d['agent_name']=='tuner'];prom+=sum(d['decision']=='PASS' for d in tun);faultprom+=sum(d['decision']=='PASS' and truth[i] for d in tun)
  g=next(r for r in summary['groups'] if (r['dataset'],r['stage'])==(ds,stage));assert (success,literal,terminal)==(g['event_success'],g['literal_terminal_success'],g['terminal_fail']);assert abs(statistics.mean(agreements)-g['sample_agreement_mean'])<1e-12;assert abs(statistics.mean(post)-g['post_fail_fraction_mean'])<1e-12
  if stage=='full':assert (prom,faultprom)==(summary['promotions'][ds]['promotions'],summary['promotions'][ds]['fault_promotions'])
  checks.append({'dataset':ds,'stage':stage,'fvr':success,'literal_release':literal,'raw_record_reconstruction':'PASS'})
# Inspect baseline override locations, not just equal aggregate scores.
base={r['seed']:r for r in records if r['stage']=='baseline_math_sentry_only'}
for stage in ['baseline_single_llm_agent','baseline_homogeneous_multi_agent']:
 for r in [x for x in records if x['stage']==stage]:
  t=json.load(gzip.open(R/r['source'],'rt'));b=json.load(gzip.open(R/base[r['seed']]['source'],'rt'));assert t['initial_fail']==b['initial_fail']
  cleared=[i for i,(x,y) in enumerate(zip(t['initial_fail'],t['final_fail'])) if x and not y];assert all(209<=i<=259 for i in cleared)
# Exact binomial discordance test recomputed by recurrence, not the original comb sum.
for t in summary['tests']:
 n=t['full_only_success']+t['baseline_only_success'];small=min(t['full_only_success'],t['baseline_only_success']);prob=2.0**(-n);total=prob
 for k in range(small):prob*= (n-k)/(k+1);total+=prob
 assert abs(min(1,2*total)-t['p'])<1e-14
# Fisher sensitivity for comparison to the professor's independent-groups analysis.
fisher=[]
for t in summary['tests']:
 successes=25-t['full_only_success'];total=25+successes
 den=math.comb(50,25);obs=math.comb(total,25)*math.comb(50-total,0)/den
 p=sum(math.comb(total,k)*math.comb(50-total,25-k)/den for k in range(max(0,25-(50-total)),min(25,total)+1) if math.comb(total,k)*math.comb(50-total,25-k)/den<=obs+1e-15)
 fisher.append({'comparison':t['comparison'],'fisher_two_sided':p,'note':'sensitivity treating groups as independent; primary test uses pairing'})
(W/'analysis/independent_verification.json').write_text(json.dumps({'checks':checks,'baseline_overrides_inside_fault':'PASS','exact_tests_second_calculation':'PASS','fisher_sensitivity':fisher},indent=2));print('Independent raw-record reconstruction passed:',len(checks),'groups; baseline locations; exact tests.')
