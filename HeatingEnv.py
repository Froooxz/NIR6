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
        # Исходные данные для моделирования
        # Нихром
        # Параметры (по паспотным данным)
        self.N_P = 1500  # Мощность нагревателя, Вт. (выбираем пока примерно для моделирования,
        # считаем, что вся электрическая мощность уходит в тепловую)
        self.N_rho = 1.1  # Удельное сопротивление нихрома, Ом*мм2/м (подбираем)
        self.N_d = 0.8  # Диаметр нихромовой проволки, мм (подбираем)

        self.N_U = 220  # Действующее напряжение сети, В (которое подаем на нагреватель)(синусоидальность напряжения рассматривать не будем,
        # в силу того, что электрические процессы гораздо быстрее рассматриваемых
        # тепловых, и моделирование приследует цели инженерных расчетов)
        self.N_R = self.N_U ** 2 / self.N_P  # Сопротивление нагревателя, Ом (высчитываем относительно проектируемой мощности)
        self.N_S = np.pi * self.N_d ** 2 / 4  # Площадь сечения нихромовой спирали, мм2 (высчитываем)
        self.N_L = self.N_R * self.N_S / self.N_rho  # Требуемая длина нихромовой спирали, м (высчитываем) (такая длина, нам теоретически нужна,
        # чтобы обеспечить тепловую мощность в 1кВт относительно выбранного
        # нихромового провода)

        self.N_c = 450  # Удельная теплоемкость нихрома, Дж/(кг*С) (по физическим свойствам)
        self.N_ro = 8300  # Плотность нихрома, кг/м3 (по физическим свойствам)

        self.N_m = self.N_ro *self. N_L * self.N_S / 1e6  # Масса нихромовой спирали, кг (расчитываем)
        self.N_F = np.pi * self.N_d / 1e3 * self.N_L  # Контактная площадь спирали, м2

        # Воздух
        # Табличные данные
        self.A_c = 1000  # Теплоемкость воздуха, Дж/(кг*С)
        self.A_ro = 1.2  # Плотность воздуха, кг/м3
        self.A_m = self.A_ro * np.pi * (0.1) ** 2 / 4 * (2)  # Масса воздуха, кг (приблизительно, при условии
        # что воздуха прокачивается через 2м трубу сечением 0,1м

        self.alf = 250  # Коэффициент теплопередачи нихром - воздух, Вт/(м2*С)

        # Электрическая часть (двигатель вентилятора)
        # Параметры определены как приблизительные, обеспечивающие схожее с
        # реальным поведение модели
        self.U = 220  # Напряжение подаваемое на вентилятор, В (Можно менять для изменения расхода воздуха)
        # изменяемый расход можно исопльзовать как возмущающее воздействие
        self.L = 1  # Индуктивность обмотки двигателя, Гн
        self.R = 100  # Активное сопротивление обмотки двигателя, Ом
        self.kw = 1  # Коэффициент скорости, отн.
        self.ke = 9  # Коэффициент самоиндукции, отн.
        self.kc = 0.037  # Коэффициент сопротивления, отн.
        # Коэффициенты подбирались приблизительно
        self.J = 0.1  # Момент инерции, кг*м2

        # Характеристики вентилятора
        self.Gnom = 200 / 3600  # Номинальный расход, м3/с
        self.wnom = 1500 / 9.54929658551369  # Номинальная частота вращения, рад/с

        # Параметры моделирования
        self.dt = 0.01  # Дискрет времени, с

        # Начальтные условия моделирования, для численного интегрирования
        self.T_a_in = 15  # Температура окружающего воздуха, С
        self.T_a = self.T_a_in
        self.T_n = self.T_a_in
        self.i = 0
        self.w = 0
        self.G_ = 0

        self.U_reg = 0  # Напряжение подаваемое на нагреватель (ЕГО МЕНЯЕМ РЕГУЛЯТОРОМ)

        #############################################################################
        self.done = False
        self.reward = np.random.uniform(-1, 1)

        # Задание целевой температуры #######
        # верхняя граница состоит из +110(нагревание на 110 при максимальной мощности)
        #  -10(если будет целевая температура T_a_in + 110 нейронная сеть сможет просто выкрутить макисмальную мощность не понимая что она делает и добьётся успеха надо давать штраф за выход за верхнюю границу)

        # self.target_temp = int(np.random.uniform(self.T_a_in, self.T_a_in + 110 - 10))  # целевая температура
        self.target_temp = 20  # целевая температура


        self.temp_error_threshold = 0.2  # Задание порогового значения погрешности температуры
        self.time_in_target_range = 0  # счетчик времени, проведенного в целевом диапазоне
        self.time_maximum_count = 0  # время моделеирования счётчик
        self.time_limit = 30  # лимит времени (время в течение которого нужно держать температуру в целевом диапазоне)
        self.time_maximum = 60  # время моделеирования (не более заданного в условии)

        self.max_temp = self.target_temp + 1000  # Максимально допустимая температура


        self.T_a_in_range = (-20, 30)
        self.U_reg_range = (0, 220)
        self.discrepancy_range = (-20, 20)
        self.T_a_range = (-20, 300)
        self.target_temp_range = (-20, 30)

        self.action_range = (0, 220)

        self.observation_space = gym.spaces.Box(low=np.array([-1, -1, -1, -1, -1]),
                                                high=np.array([1, 1, 1, 1, 1]),
                                                dtype=np.float32)  # Определение пространства состояний среды: температура и частота вращения

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(1,),
                                           dtype=np.float32)  # Определение пространства действий агента

    def step(self, action):
        rew = 0

        self.U_reg = np.interp(action[0], (-1, 1), self.action_range)  # Обновляем коэффициенты на основе действия агента

        # # Расчёт U_reg
        # if 0 > self.U_reg + action[0] or self.U_reg + action[0] > 220:
        #     pass
        # else:
        #     if self.T_a < self.target_temp:
        #         self.U_reg += abs(action[0])  # Обновляем коэффициенты на основе действия агента
        #     else:
        #         self.U_reg -= abs(action[0])  # Обновляем коэффициенты на основе действия агента


        self.hist.append(self.U_reg)


        # Уравнения электродвигателя, считаем ток и обороты
        self.i += self.dt * (1 / self.L * (self.U - self.kw * self.w - self.i * self.R))
        self.w += self.dt * (1 / self.J * (self.ke * self.i - self.kc * self.w))
        # Считаем расход воздуха относительно оборотов
        self.G_ = self.Gnom * (self.w / self.wnom)
        # Уравнения теплопередачи
        self.T_n += self.dt * (1 / (self.N_c * self.N_m) * (self.U_reg ** 2 / self.N_R - self.alf * self.N_F * (self.T_n - self.T_a)))
        self.T_a += self.dt * (1 / (self.A_c * self.A_m) * (self.alf * self.N_F * (self.T_n - self.T_a) - 2 * self.G_ * self.A_ro * self.A_c * (self.T_a - self.T_a_in)))

        discrepancy = abs(self.T_a - self.target_temp)  # невязка
        # print(discrepancy)

        self.time_maximum_count += 0.01

        ### REWARD



        # Проверяем на допустимость температуры (выход за max границу) (попытка не удачна)
        if self.T_a >= self.max_temp:
            rew -= 5  # Штраф за выход за пределы
            # self.done = True  # Считать попытку НЕ удачной и завершить
            # print('1 - выход за max границу')

        # Если долго не может достичь нужной температуры (попытка не удачна)
        if self.time_maximum_count >= self.time_maximum:
            self.done = True  # Считать попытку НЕ удачной и завершить
            # print('3 - долго не может достичь нужной температуры')

        # Если достигнута целевая температура с погрешностью и удерживается в течение времени (попытка удачна)
        if self.time_in_target_range >= self.time_limit:
            self.done = True  # Считать попытку удачной и завершить
            # print('4 - температура удерживается в течение времени')

        # Небольшая награда за удержание нужной температуры
        if discrepancy <= self.temp_error_threshold:
            rew += 0.1
            self.time_maximum_count = 0  # если достиг нужной температуры сброс счётчика (счётчик гуляния вне уставки)
            self.time_in_target_range += 0.01  # Обновляем счётчик времени в целевом диапазоне, если достигнута целевая температура
            # print('5 - удержание нужной температуры')
        else:
            self.time_in_target_range = 0  # Сбрасываем счётчик, если целевая температура не достигнута

        # Небольшая штраф за недостижения желаемой температуре
        if discrepancy >= self.temp_error_threshold:
            rew -= abs(self.T_a - self.target_temp) * 0.0005


        self.reward += rew

        observation = np.array([
            np.interp(self.T_a_in, self.T_a_in_range, (-1, 1)),
            np.interp(self.U_reg, self.U_reg_range, (-1, 1)),
            np.interp(discrepancy, self.discrepancy_range, (-1, 1)),
            np.interp(self.T_a, self.T_a_range, (-1, 1)),
            np.interp(self.target_temp, self.target_temp_range, (-1, 1))],
            dtype="object")


        return observation, self.reward, self.done, {}

    def reset(self):
        print(self.hist)
        discrepancy = abs(self.T_a - self.target_temp)

        self.done = False
        self.reward = np.random.uniform(-1, 1)

        self.time_in_target_range = 0  # счетчик времени, проведенного в целевом диапазоне
        self.time_maximum_count = 0  # максимальное время моделирования (счетчик)
        self.G_ = 0
        self.i = 0
        self.w = 0
        # self.T_a_in = int(np.random.uniform(-20, 30))  # начальная температура воздуха
        self.T_a_in = 15  # начальная температура воздуха

        self.target_temp = 20  # целевая температура
        self.max_temp = self.target_temp + 10  # обновляем верхний предел
        self.Gnom = 200/3600  # номинальный коэффициент теплоотдачи(минимум это 5% от максимума)
        self.T_n = self.T_a_in  # текущая спирали
        self.T_a = self.T_a_in  # текущая температура воздуха
        # print('T_a_in = ', self.T_a_in)
        # print('T_target = ', self.target_temp)
        # print('Gnom = ', self.Gnom)
        self.U_reg = 0

        observation = np.array([
            np.interp(self.T_a_in, self.T_a_in_range, (-1, 1)),
            np.interp(self.U_reg, self.U_reg_range, (-1, 1)),
            np.interp(discrepancy, self.discrepancy_range, (-1, 1)),
            np.interp(self.T_a, self.T_a_range, (-1, 1)),
            np.interp(self.target_temp, self.target_temp_range, (-1, 1))],
            dtype="object")

        return observation
