from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import time, csv
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, RescaleAction
from stable_baselines3 import DQN, SAC, PPO, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm
ALGOS={'dqn':DQN,'sac':SAC,'ppo':PPO,'ddpg':DDPG}
@dataclass
class TrainConfig:
    algo:str; env_id:str; timesteps:int=200000; learning_rate:float=3e-4; gamma:float=0.99;
    log_dir:str='runs/exp'; video_dir:Optional[str]=None; rescale_action:bool=False
def make_env(env_id:str, video_dir:Optional[str]=None, rescale:bool=False):
    env=gym.make(env_id, render_mode='rgb_array' if video_dir else None)
    if rescale: env=RescaleAction(env, -1, 1)
    env=RecordEpisodeStatistics(env)
    if video_dir: env=RecordVideo(env, video_folder=video_dir, name_prefix=env_id.replace('-','_')+'_agent')
    return env
def train_agent(cfg:TrainConfig):
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    env=make_env(cfg.env_id, cfg.video_dir, cfg.rescale_action)
    Algo=ALGOS[cfg.algo]
    model=Algo('MlpPolicy', env, learning_rate=cfg.learning_rate, gamma=cfg.gamma, verbose=1)
    t0=time.time(); model.learn(total_timesteps=int(cfg.timesteps)); dt=time.time()-t0
    model_path=Path(cfg.log_dir)/'model.zip'; model.save(str(model_path))
    with open(Path(cfg.log_dir)/'metrics.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['env','algo','timesteps','learning_rate','gamma','seconds']); w.writeheader();
        w.writerow({'env':cfg.env_id,'algo':cfg.algo,'timesteps':cfg.timesteps,'learning_rate':cfg.learning_rate,'gamma':cfg.gamma,'seconds':f'{dt:.2f}'})
    return str(model_path)
def eval_agent(model_path:str, env_id:str, episodes:int=10, video_dir:Optional[str]=None, rescale:bool=False):
    env=make_env(env_id, video_dir, rescale)
    model=BaseAlgorithm.load(model_path, env=env, print_system_info=False)
    rewards=[]
    for _ in range(int(episodes)):
        obs,_=env.reset(); done=False; total=0.0
        while not done:
            action,_=model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            total+=float(rew); done=terminated or truncated
        rewards.append(total)
    env.close(); return rewards
