import pickle
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def lift_term(stone_height):
    return np.clip((stone_height)/(stone_height_threshold),0.0, 1.0)

def approach_term(obs):
    bucket = to_numpy(obs["bucket_pos"]).reshape(-1)
    stone = to_numpy(obs["stone_pos"]).reshape(-1)
    dist = np.linalg.norm(bucket - stone)
    return 1.0/(1.0+dist)

def compute_reward(obs_dict, prev_obs_dict=None):
    if reward_type == "sparse":
        if to_numpy(obs_dict["stone_pos"]).reshape(-1)[2] >= stone_height_threshold:
            return 1
        else:
            return 0
    else:
        stone_height = to_numpy(obs_dict["stone_pos"]).reshape(-1)[2]

        def lift_term(stone_height):
            return np.clip(stone_height/stone_height_threshold,0.0, 1.0)

        def approach_term(obs):
            bucket = to_numpy(obs["bucket_pos"]).reshape(-1)
            stone = to_numpy(obs["stone_pos"]).reshape(-1)
            dist = np.linalg.norm(bucket - stone)
            return 1.0/(1.0+dist)
        
        if prev_obs_dict is not None:
            prev_stone_height = to_numpy(prev_obs_dict["stone_pos"]).reshape(-1)[2]
            lift_change = lift_term(stone_height) - lift_term(prev_stone_height)
            approach_change = approach_term(obs_dict) - approach_term(prev_obs_dict)
        else:
            lift_change = 0.0
            approach_change = 0.0
        
        #final success bonus
        success_bonus = 1.0 if stone_height >= stone_height_threshold else 0.0

        reward = 0.5 * lift_change + 0.3 * approach_change + success_bonus

        return reward,lift_change,approach_change,success_bonus
    
def calc_reward(obs):
    stone_pos = obs["stone_pos"]
    z = stone_pos[2]
    target_z = 1.7

    # distance to target height
    dist = z - target_z
    reward = -abs(dist)

    # If proper height reached
    if z >= 1.5:
        reward += 10

    return reward

src_dataset = 'demonstrations_reseted_env_84x84'
stone_height_threshold = 1.5
reward_type = 'dense'
gamma = 0.99
demo_dir = Path(src_dataset)
pkls = sorted(demo_dir.glob("*.pkl"))

ep_lens = []
ep_rets = []
for pkl in pkls:
    print(pkl.name)
    with pkl.open("rb") as f:
        demo = pickle.load(f)
    prev_step = None
    episode_reward = 0
    ep_lens.append(len(demo))
    for i in range(len(demo)):
        # compute rewards for this step
        if i > 0:
            prev_step = demo[i-1]
            # step_reward, step_lift_term, step_approach_term,success_bonus = compute_reward(demo[i],prev_step)
            # print(f'step: {i}, step_reward: {step_reward}, step_lift_term: {step_lift_term}, step_approach_term: {step_approach_term}, success_bonus: {success_bonus}')
            step_reward = calc_reward(demo[i])
            print(f'step: {i}, step_reward: {step_reward}')
            episode_reward += (gamma**i)*step_reward
        if i == len(demo) - 1:
            if demo[i]["stone_pos"][2] >= stone_height_threshold:
                episode_reward += 200

    print(f'Episode return = {episode_reward}')
    print()
    ep_rets.append(episode_reward)

print('max length ',max(ep_lens))
print('avg length ',sum(ep_lens)/len(ep_lens))
print('min length ',min(ep_lens))
print('min rets ',min(ep_rets))
print('max rets ',max(ep_rets))
print('avg rets ',sum(ep_rets)/len(ep_rets))


print("finished")

