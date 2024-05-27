import numpy as np
import gym


# рандомные параметры
# self.T_a_in = int(np.random.uniform(-20, 30))  # начальная температура воздуха
# self.target_temp = int(np.random.uniform(self.T_a_in, self.T_a_in + 110 - 10))  # целевая температура
# self.Gnom = np.random.uniform((250 / 3600) * 0.05, 250 / 3600)  # номинальный коэффициент теплоотдачи(минимум это 5% от максимума)

class HeatingEnv(gym.Env):
    def __init__(self):
        super(HeatingEnv, self).__init__()

        self.hist = []
        self.U = 220  # Напряжение сети для вентилятора, В
        self.N_d = 0.6  # Диаметр нихромовой проволки, мм
        self.N_S = 0.25  # Сечение нихрома, мм2
        self.N_c = 450  # Удельная теплоемкость нихрома, Дж/(кг*С)
        self.N_ro = 8300  # Плотность нихрома, кг/м3
        self.N_l = 12  # длина проводника
        ###
        self.N_R = self.N_l * 3.89  # Сопротивление нихромового нагревателя, Ом
        self.N_P = self.U ** 2 / self.N_R  # Мощность нихромового нагревателя, Вт
        self.N_m = self.N_l * self.N_S / 1e6 * self.N_ro  # масса проводника
        self.N_F = np.pi * self.N_d / 1e3 * self.N_l  # площадь поперечного сечения проводника
        ###
        self.A_c = 1000  # площадь контакта
        self.A_ro = 1.2  # площадь поперечного сечения
        self.A_m = self.A_ro * np.pi * 0.1 * 0.1 / 4  # масса
        self.alf = 500  # теплопроводность
        self.L = 1  # индуктивность
        self.R = 100  # сопротивление
        ###
        self.dt = 0.01  # Определение временного шага моделирования
        self.T_a_in = int(np.random.uniform(-20, 30))  # начальная температура воздуха
        self.T_n = self.T_a_in  # текущая спирали
        self.T_a = self.T_a_in  # текущая температура воздуха
        self.i = 0  # начальный ток
        self.w = 0  # начальная частота вращения
        self.J = 0.1  # инерция
        self.kw = 1  # коэффициент торможения
        self.ke = 9  # коэффициент нагрева
        self.kc = 0.037  # коэффициент охлаждения
        self.Gnom = np.random.uniform((250 / 3600) * 0.05,
                                      250 / 3600)  # номинальный коэффициент теплоотдачи(минимум это 5% от максимума)
        self.wnom = 2500  # номинальный коэффициент частоты вращения
        self.G_ = 0

        #############################################################################
        self.done = False
        self.reward = np.random.uniform(-1, 1)

        self.U_reg = 0  # Напряжение сети для нихрома, В 305

        # Задание целевой температуры #######
        # верхняя граница состоит из +110(нагревание на 110 при максимальной мощности)
        #  -10(если будет целевая температура T_a_in + 110 нейронная сеть сможет просто выкрутить макисмальную мощность не понимая что она делает и добьётся успеха надо давать штраф за выход за верхнюю границу)
        self.target_temp = int(np.random.uniform(self.T_a_in, self.T_a_in + 110 - 10))  # целевая температура

        self.temp_error_threshold = 1  # Задание порогового значения погрешности температуры
        self.time_in_target_range = 0  # счетчик времени, проведенного в целевом диапазоне
        self.time_maximum_count = 0  # время моделеирования счётчик
        self.time_limit = 100  # лимит времени (время в течение которого нужно держать температуру в целевом диапазоне)
        self.time_maximum = 1000  # время моделеирования (не более заданного в условии)

        self.max_temp = self.target_temp + 10  # Максимально допустимая температура

        # self.T_a_in = -20, 30
        # self.U_reg = 0, 305
        # self.T_n = -20, 2506 (наибольшая self.T_a_in и наименьший self.Gnom)
        # self.T_a = -20, 2330 (наибольшая self.T_a_in и наименьший self.Gnom)
        # self.target_temp = -20, 30 + 110 - 10

        self.T_a_in_range = (-20, 30)
        self.U_reg_range = (0, 305)
        self.T_n_range = (-20, 2506)
        self.T_a_range = (-20, 2330)
        self.target_temp_range = (-20, (30 + 110 - 10))

        self.action_range = (0, 305)

        self.observation_space = gym.spaces.Box(low=np.array([-1, -1, -1, -1, -1]),
                                                high=np.array([1, 1, 1, 1, 1]),
                                                dtype=np.float32)  # Определение пространства состояний среды: температура и частота вращения

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(1,),
                                           dtype=np.float32)  # Определение пространства действий агента

    def step(self, action):
        rew = 0

        self.U_reg = np.interp(action[0], (-1, 1), self.action_range)  # Обновляем коэффициенты на основе действия агента
        # self.U_reg += action[0]  # Обновляем коэффициенты на основе действия агента
        # self.hist.append(action[0])


        # Рассчитываем новые значения параметров
        self.i += self.dt * (1 / self.L * (self.U - self.kw * self.w - self.i * self.R))
        self.w += self.dt * (1 / self.J * (self.ke * self.i - self.kc * self.w))
        self.G_ = self.Gnom * (self.w / self.wnom)
        self.T_n += self.dt * (1 / (self.N_c * self.N_m) * (
                self.U_reg ** 2 / self.N_R - self.alf * self.N_F * (self.T_n - self.T_a)))
        self.T_a += self.dt * (1 / (self.A_c * self.A_m) * (
                self.alf * self.N_F * (self.T_n - self.T_a) - 2 * self.G_ * self.A_ro * self.A_c * (
                self.T_a - self.T_a_in)))

        discrepancy = abs(self.T_a - self.target_temp)  # невязка

        # print(discrepancy)
        self.time_maximum_count += 1

        ### REWARD

        if 0 > self.U_reg or self.U_reg > 305:
            rew -= 200
            self.done = True
            print('0 - выход U_reg за границы')

        # Проверяем на допустимость температуры (выход за max границу) (попытка не удачна)
        if self.T_a >= self.max_temp:
            rew -= 5  # Штраф за выход за пределы
            # self.done = True  # Считать попытку НЕ удачной и завершить
            # print('1 - выход за max границу')

        # Если долго не может достичь нужной температуры (попытка не удачна)
        if self.time_maximum_count == self.time_maximum:
            rew -= 20  # Штраф долгое недостижение цели
            self.done = True  # Считать попытку НЕ удачной и завершить
            # print('3 - долго не может достичь нужной температуры')

        # Если достигнута целевая температура с погрешностью и удерживается в течение времени (попытка удачна)
        if self.time_in_target_range >= self.time_limit:
            rew += 30
            self.done = True  # Считать попытку удачной и завершить
            # print('4 - температура удерживается в течение времени')

        # Небольшая награда за удержание нужной температуры
        if discrepancy <= self.temp_error_threshold:
            rew += 7
            self.time_maximum_count = 0  # если достиг нужной температуры сброс счётчика (счётчик гуляния вне уставки)
            self.time_in_target_range += 1  # Обновляем счётчик времени в целевом диапазоне, если достигнута целевая температура
            # print('5 - удержание нужной температуры')
        else:
            self.time_in_target_range = 0  # Сбрасываем счётчик, если целевая температура не достигнута

        # Небольшая штраф за недостижения желаемой температуре
        if discrepancy >= self.temp_error_threshold:
            rew -= discrepancy * 0.01 + np.random.normal(0, 0.1)

        self.reward += rew

        observation = np.array([
            np.interp(self.T_a_in, self.T_a_in_range, (-1, 1)),
            np.interp(self.U_reg, self.U_reg_range, (-1, 1)),
            np.interp(self.T_n, self.T_n_range, (-1, 1)),
            np.interp(self.T_a, self.T_a_range, (-1, 1)),
            np.interp(self.target_temp, self.target_temp_range, (-1, 1))],
            dtype="object")


        return observation, self.reward, self.done, {}

    def reset(self):
        print(self.hist)
        self.done = False
        self.reward = np.random.uniform(-1, 1)

        self.time_in_target_range = 0  # счетчик времени, проведенного в целевом диапазоне
        self.time_maximum_count = 0  # максимальное время моделирования (счетчик)
        self.G_ = 0
        self.i = 0
        self.w = 0
        self.T_a_in = int(np.random.uniform(-20, 30))  # начальная температура воздуха
        self.target_temp = int(np.random.uniform(self.T_a_in, self.T_a_in + 110 - 10))  # целевая температура
        self.max_temp = self.target_temp + 10  # обновляем верхний предел
        self.Gnom = np.random.uniform((250 / 3600) * 0.05,
                                      250 / 3600)  # номинальный коэффициент теплоотдачи(минимум это 5% от максимума)
        self.T_n = self.T_a_in  # текущая спирали
        self.T_a = self.T_a_in  # текущая температура воздуха
        # print('T_a_in = ', self.T_a_in)
        # print('T_target = ', self.target_temp)
        # print('Gnom = ', self.Gnom)
        self.U_reg = 0

        observation = np.array([
            np.interp(self.T_a_in, self.T_a_in_range, (-1, 1)),
            np.interp(self.U_reg, self.U_reg_range, (-1, 1)),
            np.interp(self.T_n, self.T_n_range, (-1, 1)),
            np.interp(self.T_a, self.T_a_range, (-1, 1)),
            np.interp(self.target_temp, self.target_temp_range, (-1, 1))],
            dtype="object")

        return observation
