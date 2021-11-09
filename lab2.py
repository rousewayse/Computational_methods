#!/bin/env python3
import numpy as np
import sys
from math import sqrt

NUM_ITERS = 7
A = np.array([[9.62483 ,1.15527,-2.97153],[1.15527,7.30891,0.69138],[-2.97153,0.69138,5.79937]], float)
B = np.array([8.71670, 5.15541, 0.27384], float)
############################
#Реализация метода Гаусса с выбором главного элемента по строке
def Gauss(matrix,  vector):
    a, b = Gauss_forward(matrix, vector)
    return Gauss_reverse(a, b)

#Прямой ход
def Gauss_forward(matrix, vector):
    n, _ = matrix.shape
    for k in range(0, n):
        #Выбираем номер строки для перестановки
        p = np.abs(matrix[k:, k]).argmax() + k
        if p != k:
            matrix[k], matrix[p] = matrix[p].copy(), matrix[k].copy()
            vector[k], vector[p] = vector[p].copy(), vector[k].copy()
        tmp = matrix[k, k]
        matrix[k, k + 1:] /= tmp
        vector[k] /= tmp

        for i in range(k + 1, n):
            tmp = matrix[i, k]
            matrix[i, k + 1:] -= matrix[k, k + 1:] * tmp
            vector[i] -= vector[k] * tmp
    return matrix, vector

#Обратный ход
def Gauss_reverse(matrix, vector):
    n, _  = matrix.shape
    res = np.array([0 for i in range(0, n)] , float)
    for i in range(n - 1, -1, -1):
        sum = np.sum(res[i + 1:] * matrix[i, i + 1:])
        res[i] = vector[i] - sum
    return res

######################

#Метод простой итерации
def simple_iteration(H_D, g_D, x0, count):
    #Список приближений x^(k)
    iters = [x0]
    
    for k in range(0, count):
        temp = H_D.dot(iters[k]) +  g_D
        iters.append( temp)
    return iters
    
def aprior_estimate(H_D, iters):
    return np.linalg.norm(H_D, np.inf) * np.linalg.norm(iters[-1] - iters[-2], np.inf) / (1 - np.linalg.norm(H_D, np.inf))

def decompose_H_D(H_D):
    H_L = np.zeros(H_D.shape, float)
    H_R = np.zeros(H_D.shape, float)
    
    for i in range(0, H_L.shape[0]):
        H_L[i][:i] = H_D[i][:i].copy();
        H_R[i][i:] = H_D[i][i:].copy()
    return H_L, H_R
#метод Зейделя
def Zeidel(H_D, g_D, x0, count):
    #Список приближений решения, начинается с x0
    iters = [x0]
    H_L, H_R = decompose_H_D(H_D)
    #EH_L = (E - H_L)^-1
    EH_L = np.linalg.inv(np.identity(H_L.shape[0], float) - H_L)
    for k in range(0, count):
        temp = EH_L@H_R.dot(iters[k]) + EH_L.dot(g_D)
        iters.append(temp)
        
    return iters

def Lusternik_speedup(H, iters):
    H_radius = np.abs(np.linalg.eigvals(H)).max()
    
    if(H_radius < 1):
        temp = iters[-2] + 1.0/(1 - H_radius)*(iters[-1] - iters[-2])
        iters.append(temp)
    return iters
#Метод верхней релаксации
def upper_relax(H_D, g_D, x0, count):
    iters = [x0]
    H_radius = np.abs(np.linalg.eigvals(H_D)).max()
    qopt = 2.0/(1 + sqrt(1 - H_radius**2))
    
    for k in range(0, count):
        temp = np.zeros(x0.shape, float)
        
        for i in range(0, x0.shape[0]):
            s = 0
            for j in range(0, i):
                s += H_D[i][j]*temp[j]
            for j in range(i+1, x0.shape[0]):
                s += H_D[i][j]*iters[k][j]
            s += g_D[i] - iters[k][i]
            temp[i] = iters[k][i] + qopt*s
        iters.append(temp)
    return iters


#Преобразование системы к необходимому виду
def transform_equation(matrix, vector):
    #Определим матрицу D
    D = np.identity(matrix.shape[0], float)
    for i in range(0, matrix.shape[0]):
        D[i][i] = matrix[i][i];
    #Определим матрицу H_{D}
    H_D = np.identity(matrix.shape[0], float) - np.linalg.inv(D).dot(matrix)
    #Определим вектор g_{D}
    g_D = np.linalg.inv(D).dot(vector)
    
    return np.array([0.0 for i in range(0, matrix.shape[0])], float), H_D, g_D




acc_sol = Gauss(A.copy(), B.copy())

print(f"Найдем точное решение x* методом Гаусса: {acc_sol}")

x0, H_D, g_D = transform_equation(A, B)
print("Выполним преобразование системы к нужному виду: ")
print("H_D = ")
print(H_D)
print("g_D = ")
print(g_D)
print("Нулевое приближение x0 положим ")
print(x0)

norm_H_D = np.linalg.norm(H_D, ord=np.inf)
print(f"Вычиcлим ||H_D||_infty = {norm_H_D}" )

iters = simple_iteration(H_D, g_D, x0, 7)
print("Вычислим методом простой итерации 7-е приближение решения x^(7): ")
print(iters[-1])

print(f"Апостериорная оценка: {aprior_estimate(H_D, iters)}")

def act_err(acc_sol, iters):
    return np.linalg.norm(acc_sol - iters[-1], np.inf) / np.linalg.norm(acc_sol, np.inf)

print("Сравним апостеорную оценку с фактической погрешностью:")
print(f"||x* - x^(7)||_infty = {aprior_estimate(H_D, iters)} \n delta_x = {act_err(acc_sol,  iters)} ")

print("Матрицу H_D представим в виде H_L + H_R:")
H_L, H_R = decompose_H_D(H_D)
print("H_L = ")
print(H_L)
print("H_R = ")
print(H_R)
iters = Zeidel(H_D, g_D, x0, NUM_ITERS)
print("Вычислим методом Зейделя 7-е приближение решения x^(7): ")
print(iters[-1])

print(f"Фактическая погрешность метода Зейделя: {act_err(acc_sol, iters)}")
zeidel_H = np.linalg.inv(np.identity(H_L.shape[0], float) - H_L)@H_R
print(f"Спектральный радиус матрицы перехода метода Зейделя: {np.abs(np.linalg.eigvals(zeidel_H)).max()}")


iters = Lusternik_speedup(zeidel_H, iters)
print("Последнее приблежение метода Зейделя после уточнения по Люстернику:")
print(iters[-1])

print("Фактическая погрешность после уточнения: ")
print(act_err(acc_sol, iters))

print("Вычислим методом верхней релаксации 7-е приближение решения x^(7): ")
iters = upper_relax(H_D, g_D, x0,  NUM_ITERS)
print(iters[-1])
print("Фактическая погрешность метода верхней релаксации:")
print(act_err(acc_sol, iters))

