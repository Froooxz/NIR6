import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from HeatingEnv import HeatingEnv


# Загрузка обученной модели
model = DDPG.load('model')

# Инициализация переменных для хранения значений
T_a = []
U_reg = []
rewards = []  # Список для хранения значений reward
target = []
G_ = []
env = HeatingEnv()
env = DummyVecEnv([lambda: env])

# Сброс среды
obs = env.reset()
i=0
sbros = 0
done = False
# Запуск предсказаний на каждом временном шаге моделирования
while sbros < 1:

    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Получение предсказанного действия от модели
        obs, reward, done, _ = env.step(action)  # Выполнение действия в среде

        # Сохранение T_a U_reg reward
        T_a.append(env.envs[0].T_a)
        target.append(env.envs[0].target_temp)
        U_reg.append(env.envs[0].U_reg)
        G_.append(env.envs[0].Gnom + env.envs[0].Gnom_noise[int(i)])

        rewards.append(reward)
        i += 1
    done = False
    sbros += 1

print(G_)
# Преобразование U_reg к одномерному массиву
U_reg_flat = [x[0] if isinstance(x, np.ndarray) else x for x in U_reg]
U_reg_flat = np.array(U_reg_flat)

# Построение графика
plt.figure(figsize=(10, 9))

plt.subplot(4, 1, 1)

l = len(T_a[::1])
plt.plot(T_a[::1], label='T_a')
plt.plot(target[::1], label='target')
plt.plot([target[0]-env.envs[0].temp_error_threshold]*l, color='C5',  alpha=0.6)
plt.plot([target[0]+env.envs[0].temp_error_threshold]*l, color='C5',  alpha=0.6)
plt.ylabel('T')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(U_reg_flat[::1])
plt.ylabel('U_reg')
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(rewards)
plt.ylabel('Reward')
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(G_)
plt.ylabel('Gnom + noise')
plt.grid()


plt.show()
