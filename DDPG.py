from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


import numpy as np
from HeatingEnv import HeatingEnv

env = HeatingEnv()
env = DummyVecEnv([lambda: env])


# Обучение с использованием DDPG
model = DDPG(
    policy=MlpPolicy,  # используем MlpPolicy в качестве политики
    env=env,  # среда обучения
    # learning_rate=0.01,  # скорость обучения
    # buffer_size=1000000,  # размер буфера воспроизведения
    # batch_size=128,  # размер пакета
    # learning_starts=1000,  # обучение после выполнения X шагов
    # train_freq=1,  # обновляем политику каждые X шагов
    # gradient_steps=10,  # количество шагов градиентного спуска для обновления политики
    policy_kwargs=dict(net_arch=[8]),  # настройка архитектуры нейронной сети политики
    verbose=1  # информацию о ходе обучения
)

model.learn(total_timesteps=1000)


# Сохранение модели

model.save('model')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)


print('###########################')
print(mean_reward)
print(std_reward)


