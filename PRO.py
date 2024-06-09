from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from PROEnv import HeatingEnv

# Создание среды
env = HeatingEnv()
env = DummyVecEnv([lambda: env])

# Создание модели PPO
model = PPO(
    policy='MlpPolicy',  # используем MlpPolicy в качестве политики
    env=env,  # среда обучения
    verbose=1,  # вывод информации о ходе обучения
    policy_kwargs=dict(net_arch=[8, 8]),  # настройка архитектуры нейронной сети политики
)

# Обучение модели
model.learn(total_timesteps=1000)

# Сохранение модели
model.save('ppo_model')

# Оценка модели
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

print('###########################')
print(mean_reward)
print(std_reward)
