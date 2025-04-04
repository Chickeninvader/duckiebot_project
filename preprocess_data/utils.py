def compute_reward(d, phi, velocity, in_lane):
    """Compute reward based on lane position, heading, velocity, and whether in lane"""
    # Reference values
    d_ref = 0.0  # Ideal lateral offset
    phi_ref = 0.0  # Ideal heading angle

    # Tunable weights
    alpha = 1.0     # Weight for lateral offset penalty
    beta = 0.0  # No penalty for heading angle in this case
    gamma = 0.0  # No reward for velocity in this case
    lambda_ = 100.0  # High penalty for leaving the lane

    # Compute reward components
    r_d = -alpha * abs(d - d_ref) * abs(d - d_ref)
    r_phi = -beta * abs(phi - phi_ref)
    r_v = gamma * velocity  # Reward movement
    r_lane = -lambda_ if not in_lane else 1  # High penalty for being out of lane and encouragement for being in lane

    reward = r_d + r_phi + r_v + r_lane
    return reward


"""
new reward function with fuel based calculation.
this new function uses fuel_consumption as an additional parameter.
The penalty for fuel consumption is set at 0.01, which means for every unit of fuel consumed, there is a -0.01 penalty to the reward.
"""


def compute_reward_fuel(d, phi, velocity, in_lane, fuel_consumption):
    """Compute reward based on lane position, heading, velocity, in lane status and fuel consumption"""
    # Reference values
    d_ref = 0.0  # Ideal lateral offset
    phi_ref = 0.0  # Ideal heading angle

    # Tunable weights
    alpha = 1.0
    beta = 0.1
    gamma = 1.0
    lambda_ = 5.0  # High penalty for leaving the lane
    fuel_penalty = 0.01  # Penalty for fuel consumption

    # Compute reward components
    r_d = -alpha * abs(d - d_ref)
    r_phi = -beta * abs(phi - phi_ref)
    r_v = gamma * velocity  # Reward movement
    r_lane = -lambda_ if not in_lane else 0  # High penalty for being out of lane
    r_fuel = -fuel_penalty * fuel_consumption  # Penalty for fuel consumption

    # Total reward
    if velocity <= 0.05:
        return r_d + r_phi + (gamma if in_lane else -lambda_) + r_fuel

    reward = r_d + r_phi + r_v + r_lane + r_fuel
    return reward
