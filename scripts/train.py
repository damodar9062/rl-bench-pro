import argparse
from rl_bench_pro.train import TrainConfig, train_agent
p=argparse.ArgumentParser(); p.add_argument('--algo',choices=['dqn','sac','ppo','ddpg'],required=True); p.add_argument('--env',required=True);
p.add_argument('--timesteps',type=int,default=200000); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--gamma',type=float,default=0.99);
p.add_argument('--log',default='runs/exp'); p.add_argument('--video',default=None); p.add_argument('--rescale',action='store_true');
args=p.parse_args(); cfg=TrainConfig(algo=args.algo, env_id=args.env, timesteps=args.timesteps, learning_rate=args.lr, gamma=args.gamma, log_dir=args.log, video_dir=args.video, rescale_action=args.rescale); mp=train_agent(cfg); print('Saved model to:', mp)
