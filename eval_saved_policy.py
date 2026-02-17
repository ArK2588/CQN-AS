import os
import argparse
import json
from pathlib import Path
import yaml
from types import SimpleNamespace
import numpy as np
import torch

# Match training runtime env setup
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import agxcave.agxtasks
import agx_src.agx_env as agx_env
import utils
from video import VideoRecorder


def load_policy(policy_path: Path, device: torch.device):
    
    payload = torch.load(policy_path, map_location=device, weights_only=False)

    if "agent" not in payload:
        raise KeyError(f"No 'agent' key in policy checkpoint: {policy_path}")

    agent = payload["agent"]
    global_step = int(payload.get("_global_step", 0))
    global_episode = int(payload.get("_global_episode", 0))

    agent.device = device
    if hasattr(agent, "encoder"):
        agent.encoder.to(device)
    if hasattr(agent, "critic"):
        agent.critic.to(device)
    if hasattr(agent, "critic_target"):
        agent.critic_target.to(device)
        agent.critic_target.eval()
    if hasattr(agent, "train"):
        agent.train(False)

    return agent, global_step, global_episode


def build_cfg(config_path: Path, num_eval_episodes=None, device=None):
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    keys = [
        "task_name",
        "episode_length",
        "frame_stack",
        "dataset_root",
        "camera_shape",
        "stone_height_threshold",
        "camera_keys",
        "state_based_only",
        "reward_type",
        "reward_config",
        "temporal_ensemble",
        "action_sequence",
        "action_repeat",
        "num_eval_episodes",
        "device",
    ]
    cfg_dict = {k: raw[k] for k in keys}
 
    if num_eval_episodes is not None:
        cfg_dict["num_eval_episodes"] = int(num_eval_episodes)
    if device is not None:
        cfg_dict["device"] = str(device)
 
    return SimpleNamespace(**cfg_dict)


def evaluate(agent, cfg, global_step, output_dir: Path, save_video: bool, headless: bool, render_mode=None):
    env = agx_env.make(
        cfg.task_name,
        cfg.episode_length,
        cfg.frame_stack,
        cfg.dataset_root,
        cfg.camera_shape,
        cfg.stone_height_threshold,
        camera_keys=cfg.camera_keys,
        state_based_only=cfg.state_based_only,
        reward_type=cfg.reward_type,
        reward_config=cfg.reward_config,
        headless=headless,
        render_mode=render_mode,
    )

    eval_temporal_ensemble = None
    if cfg.temporal_ensemble:
        eval_temporal_ensemble = utils.TemporalEnsembleControl(
            cfg.episode_length,
            env.action_spec(),
            cfg.action_sequence,
        )

    recorder = VideoRecorder(output_dir if save_video else None)

    step = 0
    episode = 0
    total_reward = 0.0

    num_rock_lifted = 0
    num_fall_down = 0
    num_success = 0

    terminations_info = {
        "max_steps": 0,
        "too_deep_termination": 0,
        "stone_x_distance_termination": 0,
        "stone_height_termination": 0,
        "cabin_pitch_termination": 0,
    }
    end_positions = []
    rewards_info = {}

    eval_until_episode = utils.Until(cfg.num_eval_episodes)

    while eval_until_episode(episode):
        episode_step = 0
        episode_rock_lifted = False
        episode_success = False
        episode_fall_down = False

        time_step = env.reset()
        if eval_temporal_ensemble is not None:
            eval_temporal_ensemble.reset()

        recorder.init(env, enabled=(episode == 0))

        while not time_step.last():
            if cfg.temporal_ensemble or (episode_step % cfg.action_sequence == 0):
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(
                        time_step.rgb_obs,
                        time_step.low_dim_obs,
                        global_step,
                        eval_mode=True,
                    )
                action = action.reshape([cfg.action_sequence, -1])
                if eval_temporal_ensemble is not None:
                    eval_temporal_ensemble.register_action_sequence(action)

            if eval_temporal_ensemble is not None:
                sub_action = eval_temporal_ensemble.get_action()
            else:
                sub_action = action[episode_step % cfg.action_sequence]

            time_step = env.step(sub_action)
            recorder.record(env)

            total_reward += float(time_step.reward)

            # stone position checks
            stone_z = float(env._last_stone_pos[2])
            if stone_z >= 1.5:
                episode_rock_lifted = True
            if episode_rock_lifted and stone_z <= 1.0:
                episode_fall_down = True

            # success via rock_stable reward
            rock_stable = env._last_step_rewards.get("rock_stable", 0)
            if rock_stable != 0:
                episode_success = True

            for key, value in env._last_step_rewards.items():
                rewards_info.setdefault(key, []).append(float(value))

            step += 1
            episode_step += 1

        end_positions.append(float(env._last_stone_pos[2]))
        if episode_rock_lifted:
            num_rock_lifted += 1
        if episode_success:
            num_success += 1
        if episode_fall_down:
            num_fall_down += 1

        for termination_type, triggered in env._last_termination_info.items():
            if triggered:
                terminations_info[termination_type] = terminations_info.get(termination_type, 0) + 1

        recorder.save(f"{global_step}_ep{episode}.mp4")
        episode += 1

    # Aggregate metrics (mirrors train eval loop)
    summary = {
        "episode_reward": total_reward / episode,
        "episode_length": step * cfg.action_repeat / episode,
        "episode": episode,
        "step": global_step,
        "rock_lifted_ratio": num_rock_lifted / episode,
        "success_ratio": num_success / episode,
        "fall_down_ratio": num_fall_down / episode,
        "mean_end_position": float(np.mean(end_positions)),
    }

    for termination_type, count in terminations_info.items():
        summary[f"term/{termination_type}"] = count / episode

    for key, values in rewards_info.items():
        if len(values) > 0:
            summary[f"reward/{key}"] = float(np.mean(values))

    return summary


def main():
    parser = argparse.ArgumentParser("Evaluate saved CQN-AS AGX policy")
    parser.add_argument("--policy-path", type=Path, required=True, help="Path to policy_XXXX.pt")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("cfgs/config_cqn_as_agx.yaml"),
        help="Path to config yaml",
    )
    parser.add_argument("--num-eval-episodes", type=int, default=None, help="Override cfg.num_eval_episodes")
    parser.add_argument("--device", type=str, default=None, help="Override cfg.device (e.g. cuda, cpu)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-video", action="store_true", help="Save first episode video")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Where eval_video/ is written")
    parser.add_argument("--results-json", type=Path, default=None, help="Optional json output file")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-mode",type=str,default="none",help="AGX render mode")
    args = parser.parse_args()

    cfg = build_cfg(
        args.config_path,
        num_eval_episodes=args.num_eval_episodes,
        device=args.device,
    )

    utils.set_seed_everywhere(args.seed)
    device = torch.device(cfg.device)

    agent, global_step, global_episode = load_policy(args.policy_path, device)
    print(f"Loaded policy from {args.policy_path}")
    print(f"_global_step={global_step}, _global_episode={global_episode}, device={device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_mode = None if args.render_mode == "none" else args.render_mode
    summary = evaluate(agent, cfg, global_step, args.output_dir, args.save_video,args.headless,render_mode)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with args.results_json.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved metrics to {args.results_json}")


if __name__ == "__main__":
    main()