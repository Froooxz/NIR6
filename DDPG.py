from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



from HeatingEnv import HeatingEnv


import torch.nn as nn

class CustomMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_critics):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Первый скрытый слой с 64 нейронами
        self.fc2 = nn.Linear(64, 64)         # Второй скрытый слой с 64 нейронами
        self.fc3 = nn.Linear(64, output_dim) # Выходной слой с заданным числом выходных нейронов

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))  # Применение линейной активации ReLU к первому скрытому слою
        x = nn.functional.relu(self.fc2(x))  # Применение линейной активации ReLU ко второму скрытому слою
        x = self.fc3(x)                      # Получение выхода
        return x


env = HeatingEnv()
env = DummyVecEnv([lambda: env])


# Обучение с использованием DDPG
model = DDPG(
    policy=CustomMLP,  # используем MlpPolicy в качестве политики
    env=env,  # среда обучения
    # learning_rate=0.01,  # скорость обучения
    # buffer_size=1000000,  # размер буфера воспроизведения
    # batch_size=128,  # размер пакета
    # learning_starts=1000,  # обучение после выполнения X шагов
    # train_freq=1,  # обновляем политику каждые X шагов
    # gradient_steps=10,  # количество шагов градиентного спуска для обновления политики
    # policy_kwargs=dict(n_critics=1),  # Пример настройки архитектуры нейронной сети политики
    verbose=1  # информацию о ходе обучения
)

model.learn(total_timesteps=1000)


# Сохранение модели
model.save('model')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)


print('###########################')
print(mean_reward)
print(std_reward)


