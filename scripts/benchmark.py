import argparse, pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--inputs',nargs='+',required=True); p.add_argument('--out',default='runs/summary.csv'); args=p.parse_args();
dfs=[]
for path in args.inputs:
    try: dfs.append(pd.read_csv(path))
    except Exception: pass
import sys
if not dfs: print('No metrics parsed.'); sys.exit(0)
df=pd.concat(dfs, ignore_index=True); print(df); df.to_csv(args.out, index=False); print('Saved', args.out)
