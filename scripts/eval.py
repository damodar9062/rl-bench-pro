import argparse, json
from pathlib import Path
from rl_bench_pro.train import eval_agent
p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--env',required=True); p.add_argument('--episodes',type=int,default=10);
p.add_argument('--video',default=None); p.add_argument('--rescale',action='store_true'); p.add_argument('--out',default=None);
args=p.parse_args(); rewards=eval_agent(args.model, args.env, args.episodes, args.video, args.rescale); print('Rewards:', rewards);
if args.out: Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps({'rewards':rewards,'mean':float(sum(rewards)/len(rewards))}, indent=2), encoding='utf-8'); print('Wrote', args.out)
