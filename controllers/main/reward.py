import numpy as np


ACTION_RATE_WEIGHT = 0.01
ALIVE_REWARD = 0.2
VERTICAL_VELOCITY_WEIGHT = 0.3
VERTICAL_CHANGE_WEIGHT = 0.2

def calculate_reward(
        local_velocity,
        angular_velocity,
        upright,
        body_height,
        target_height,
        fall_threshold,
        upright_threshold,
        fallen,
        current_action,
        previous_action,
        previous_vertical_velocity
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

    #o quanto da altura ideal estamos atingindo (debug)
    height_ratio = body_height / target_height

    #controla como iremos colocar o ratio de altura na reward, de forma cubica, quadratica e etc, usando uma curva gaussiana para evitar recompensas ao pular alto demais.
    curve_radius = 0.15 
    height_factor = np.exp(-((body_height - target_height) / curve_radius) ** 2)

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

    # penaliza pouco movimentos verticais pequenos, mas movimentos grandes aumentam drasticamente
    vertical_penalty = (VERTICAL_VELOCITY_WEIGHT * abs(vertical_velocity) ** 2)

    # Penaliza mudanças bruscas na velocidade vertical
    # especialmente util para impactos/rebotes
    vertical_velocity_difference = (vertical_velocity - previous_vertical_velocity)

    vertical_change_penalty = (
        VERTICAL_CHANGE_WEIGHT
        * vertical_velocity_difference ** 2
    )


    rotation_penalty = (
        0.03 * np.linalg.norm(angular_velocity)
    )

    action_difference = (
        current_action
        - previous_action
    )

    action_rate_penalty = (
        ACTION_RATE_WEIGHT
        * np.sum(action_difference ** 2)
    )

    reward = (
        ALIVE_REWARD
        + forward_reward
        - vertical_penalty
        - vertical_change_penalty
        - rotation_penalty
        - action_rate_penalty
    )

    if fallen:
        reward -= 15.0
        
    reward_info = {
        "stability_factor": float(stability_factor),
        "height_ratio": float(height_ratio),
        "height_factor": float(height_factor),
        "forward_reward": float(forward_reward),
        "vertical_penalty": float(vertical_penalty),
        "vertical_change_penalty": float(vertical_change_penalty),
        "rotation_penalty": float(rotation_penalty),
        "action_rate_penalty": float(action_rate_penalty)
    }

    return float(reward), reward_info