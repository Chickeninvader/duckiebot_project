
def compute_reward(d, phi, velocity, in_lane):
    """Compute reward based on lane position, heading, velocity, and whether in lane"""
    # Reference values
    d_ref = 0.0  # Ideal lateral offset
    phi_ref = 0.0  # Ideal heading angle

    # Tunable weights
    alpha = 1.0
    beta = 0.5
    gamma = 1.0
    lambda_ = 5.0  # High penalty for leaving the lane

    # Compute reward components
    r_d = -alpha * abs(d - d_ref)
    r_phi = -beta * abs(phi - phi_ref)
    r_v = gamma * velocity  # Reward movement
    r_lane = -lambda_ if not in_lane else 0  # High penalty for being out of lane

    # Total reward
    if velocity <= 0.05:
        return r_d + r_phi + (gamma if in_lane else -lambda_)

    reward = r_d + r_phi + r_v + r_lane
    return reward
