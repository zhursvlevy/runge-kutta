import sys
import numpy as np
from typing import Callable, Iterable


class Cancer:
    """нормальная система дифференциальных уравнений
       описывает эволюцию раковых клеток и лейкоцитов с течением времени
    """
    def __init__(self, l1=1, 
                 l2=1, beta1=1,    
                 beta2=3, c=3) -> None:
        
        self.l1 = l1
        self.l2 = l2
        self.beta1 = beta1
        self.beta2 = beta2
        self.c = c

    def __call__(self, time: float, X: np.array) -> np.array:
        
        dX0 = (-self.l1 + self.beta1 * X[1] ** (2./3) * (1 - X[0]/self.c) * (1 + X[0])) * X[0]
        dX1 = self.l2 * X[1] - self.beta2 * X[0] * X[1]**(2./3) / (1 + X[0])

        return np.array([dX0, dX1], dtype=np.float64)

def fixed_integrating_step(system, X, time, step):
    K1 = system(time, X) *step
    K2 = system(time + step/2, X + step/2) * step
    K3 = system(time + step, X - K1 + 2*K2) * step
    time += step
    return time, X + 1/6*(K1 + 4*K2 + K3)


def variative_integrating_step(system, X, time, step, rtol=1e-3):
    N = 1
    cond = True
    while cond:
        XN = X.copy()
        X2N = X.copy()
        timeN = time
        time2N = time
        for i in range(N):
            timeN, XN = fixed_integrating_step(system, XN, timeN, step)
        for i in range(2*N):
            time2N, X2N = fixed_integrating_step(system, X2N, time2N, step/2)

        if np.linalg.norm(X2N - XN, ord=np.inf)/7 < rtol:
            cond = False
            # print(np.linalg.norm(X2N - XN))

            return XN

        else:
            step /= 2
            N = N**2
        

def RK3(system: Callable, 
        init_value: np.array, 
        interval: tuple,
        step: float = 0.01,
        variative_step=False):
    """
        Метод Рунге_Кутты 3-го порядка
        system: правая часть нормальной системы вида F(t, y), 
        где y - вектор
        init_value: начальные условия, вектор
        interval: отрезок интегрирования
        step: шаг интегрирования
        variative_step: False по умолчанию, если присвоено False,
        то интегрирование происходит с постоянным шагом, иначе с
        переменным по правилу Рунге
    """
    X = init_value
    time = interval[0]
    solution = [X]
    time_list = [time]

    if variative_step:
        while time <= interval[1]:
            X = variative_integrating_step(system, X, time, step)
            time += step
            solution.append(X)
            time_list.append(time)
    else:
        while time <= interval[1]:
            time, X = fixed_integrating_step(system, X, time, step)
            solution.append(X)
            time_list.append(time)

    return np.array(time_list, dtype=np.float64), np.array(solution, dtype=np.float64).T        

