import numpy as np
import matplotlib.pyplot as plt

import numpy as np


U_reg = 305  # Напряжение сети для нихрома, В
U = 220  # Напряжение сети для вентилятора, В
N_d = 0.6  # Диаметр нихромовой проволки, мм
N_S = 0.25  # Сечение нихрома, мм2
N_c = 450  # Удельная теплоемкость нихрома, Дж/(кг*С)
N_ro = 8300  # Плотность нихрома, кг/м3
N_l = 12  # длина проводника
###
N_R = N_l * 3.89  # Сопротивление нихромового нагревателя, Ом
N_P = U ** 2 / N_R  # Мощность нихромового нагревателя, Вт
N_m = N_l * N_S / 1e6 * N_ro  # масса проводника
N_F = np.pi * N_d / 1e3 * N_l  # площадь поперечного сечения проводника
###
A_c = 1000  # площадь контакта
A_ro = 1.2  # площадь поперечного сечения
A_m = A_ro * np.pi * 0.1 * 0.1 / 4  # масса
alf = 500  # теплопроводность
L = 1  # индуктивность
R = 100  # сопротивление
###
dt = 0.01  # Определение временного шага моделирования
t_end = 15  # Время окончания моделирования
t = np.arange(0, t_end + dt, dt)  # Создание массива временных шагов от 0 до t_end с шагом dt
T_a_in = int(np.random.uniform(5, 35))  # начальная температура воздуха

J = 0.1  # инерция
kw = 1  # коэффициент торможения
ke = 9  # коэффициент нагрева
kc = 0.037  # коэффициент охлаждения
Gnom = 250 / 3600  # номинальный коэффициент теплоотдачи
wnom = 1500  # номинальный коэффициент частоты вращения

#############################################################################

i = np.zeros(len(t))
i[0] = np.random.uniform(0, 0.64094)
w = np.zeros(len(t))
w[0] = int(np.random.uniform(0, 1488))
G_ = np.zeros(len(t))
T_n = np.zeros(len(t)) + T_a_in
T_a = np.zeros(len(t)) + T_a_in

for j in range(int(t_end / dt)):
    i[j + 1] = i[j] + dt * (1 / L * (U - kw * w[j] - i[j] * R))
    w[j + 1] = w[j] + dt * (1 / J * (ke * i[j] - kc * w[j]))
    G_[j + 1] = Gnom * (w[j + 1] / wnom)
    T_n[j + 1] = T_n[j] + dt * (1 / (N_c * N_m) * (U_reg ** 2 / N_R - alf * N_F * (T_n[j] - T_a[j])))
    T_a[j + 1] = T_a[j] + dt * (
                1 / (A_c * A_m) * (alf * N_F * (T_n[j] - T_a[j]) - 2 * G_[j] * A_ro * A_c * (T_a[j] - T_a_in)))

w = w * 9.54929658551369
print(i[0])
print(w[0])
print(T_a_in)
plt.figure(1)
plt.plot(t, T_n, t, T_a)
plt.grid(True)
plt.legend(['T_n', 'T_a'])
plt.figure(2)
plt.plot(t, w)
plt.grid(True)
plt.show()
print(T_n[0], T_n[-1])
print(T_a[0], T_a[-1])
