import gymnasium as gym


def evaluate(
    env: gym.Env,
    agent,
    num_episodes: int=1
):
    
    total_rewards = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            planned_actions = agent(obs=obs)
            action = planned_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        total_rewards.append(total_reward)

    return total_rewards