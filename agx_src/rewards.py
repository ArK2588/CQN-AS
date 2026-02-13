import torch

def calc_reward1(obs, last):
    stone = obs["stone"]
    bucket = obs["bucket"]
    z = stone[2]
    target_z = 1.7

    # --- 1. Distance to target height ---
    dist_height = z - target_z
    reward = 2 * -abs(dist_height) / 60.0

    # --- 2. Bonus for reaching proper height ---
    if z >= 1.5:
        reward += 0.5

    # --- 3. Extra final bonus ---
    if last and z >= 1.5:
        reward += 50.0

    # --- 4. Distance-based proximity reward ---
    # Euclidean distance between bucket and stone
    bucket_tensor = torch.tensor(bucket)
    stone_tensor = torch.tensor(stone)
    dist_proximity = torch.norm(bucket_tensor - stone_tensor)
    # Reward: je kleiner der Abstand, desto größer der Reward
    epsilon = 1e-6
    proximity_reward = 1.0 / (dist_proximity + epsilon)

    # Kombiniere die Rewards
    reward += float(proximity_reward)

    return reward


def calc_reward2(obs, last):
    stone = obs["stone"]
    z = stone[2]
    target_z = 1.7

    # distance to target height
    dist = z - target_z
    reward = 2 * -abs(dist)/60.0

    # If proper height reached
    if z >= 1.5:
        reward += 0.5

    if last and z >= 1.5:
        reward += 50.0

    return reward


def calc_reward3(obs, last):
    z = obs["stone"][2]

    reward = 0.0

    # --- rock_stable---
    if last and z >= 1.5:
        reward += 50.0

    # --- rock_height_clipped_reward ---
    max_z = 1.7
    height_reward = max(0.0, min(z / max_z, 1.0))
    reward += 0.5 * height_reward

    return reward


def calc_reward4(obs, last):
    z = obs["stone"][2]

    reward = 0.0

    # --- rock_stable ---
    if last and z >= 1.5:
        reward += 50.0

    # --- rock_over_1_5_reward ---
    if z >= 1.5:
        reward += 0.5

    return reward


def calc_reward5(obs, last):
    z = obs["stone"][2]
    reward = 0.0

    if last and z >= 1.5:
        reward += 50.0

    return reward