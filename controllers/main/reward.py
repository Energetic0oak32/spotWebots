import numpy as np


def calculate_reward(
        local_velocity,
        angular_velocity,
        upright,
        body_height,
        target_height,
        fall_threshold,
        upright_threshold,
        fallen,
    ):

    #pega o valor entre o threshold de queda e a inclinação perfeita e converte no intervalo de 0 a 1
    #ex: 0.78 ----- 1.00  vira algo entre 0 e 1, então 0.85 viraria 0.5 no fator de estabilidade
    stability_factor = (
        (upright - fall_threshold)
        / (upright_threshold - fall_threshold)
    )

    #apenas impede valores maiores que 1 ou menores que 0
    stability_factor = np.clip(
        stability_factor,
        0.0,
        1.0
    )

    #o quanto da altura ideal estamos atingindo
    height_ratio = body_height / target_height

    #controla como iremos colocar o ratio de altura na reward, de forma cubica, quadratica e etc...
    height_factor = np.clip(
        height_ratio ** 2,
        0.0,
        1.0
    )

    forward_velocity = local_velocity[0]
    vertical_velocity = local_velocity[2]

    if forward_velocity >= 0.0:
        forward_reward = (
            forward_velocity
            * stability_factor
            * height_factor
        )
    else:
        forward_reward = forward_velocity

    vertical_penalty = 0.15 * abs(vertical_velocity)

    rotation_penalty = (
        0.03 * np.linalg.norm(angular_velocity)
    )

    reward = (
        forward_reward
        - vertical_penalty
        - rotation_penalty
    )

    if fallen:
        reward -= 10.0
        
    reward_info = {
        "stability_factor": float(stability_factor),
        "height_ratio": float(height_ratio),
        "height_factor": float(height_factor),
        "forward_reward": float(forward_reward),
        "vertical_penalty": float(vertical_penalty),
        "rotation_penalty": float(rotation_penalty),
    }

    return float(reward), reward_info