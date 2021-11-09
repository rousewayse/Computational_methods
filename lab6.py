#!/bin/env python3
import numpy as np
import pandas as pd
from math import *
from scipy.integrate import quad
import matplotlib.pyplot as plt

#Многочлены Якоби
def Jpoly(arg, n, k = 1):
    if n < 0:
        return 0
    poly = [1, (k+1)*arg]
    for i in range (2, n+1):
        temp = ((i+k)*(2*i+2*k-1)*arg*poly[i-1] - (i+k)*(i+k-1)*poly[i-2])/((i+2*k)*(i))
        poly.append(temp)
    return poly[n]
#n>=1
def dJpoly(arg, n, k = 1):
    if n < 1:
        return 0
    return  (n+2*k+1)/2*Jpoly(arg, n-1, k+1)
def ddJpoly(arg, n, k = 1):
    if n < 2:
        return 0
    return (n + 2*k+1)*((n-1)+2*(k+1)+1)/4*Jpoly(arg, n-2, k+2)

#Координатные функции
def omega(arg, n):
    k = 1
    return (1 - arg**2)*Jpoly(arg, n, 1)
#k>=1
#Производные координатных функий
def domega(arg, n):
    k = 1
    return -2*(n+1)*Jpoly(arg, n+1, k-1)

def ddomega(arg, n):
    k = 1
    return -2*(n+1)*(n+1 +2*(k-1)+1)/2*Jpoly(arg, n, k)

def p(x):
    return -(4-x)/(5-2*x)
def q(x):
    return +(1 - x)/2
def r(x):
    return 1/2*log(3+x)
def f(x):
    return 1 - x/3

#u(-1) = u(1) = 0
def Lomega(j):
    return lambda arg: p(arg)*ddomega(arg, j) + q(arg)*domega(arg,j) + r(arg)*omega(arg, j)

#Метод Галеркина
def Galerkin(n):
    def build_eq():
        F = [quad(lambda arg: f(arg)*omega(arg, i), -1, 1)[0] for i in range(1, n+1)]
        #for i in range(1, n+1):
        #    F.append(quad(lambda arg: f(arg)*omega(arg, i), -1, 1))
        def buildAi(i):
           return   [quad( lambda arg: Lomega(j)(arg)*omega(arg,i), -1, 1)[0] for j in range(1, n+1)]
        A = [buildAi(i) for i in range(1, n+1)]
        return A, F    
    A, F = build_eq()
    C = np.linalg.solve(A, F)
    return A, F, C
#Вычисление приближенного решения
def build_solution( c):
    def u(arg):
        sum = 0
        for i in range(0, len(c)):
            sum += c[i]*omega(arg, i+1)
        return sum
    return u
#Корни многочлена чебышева
def Chebyshev_roots(n):
    return [cos((2*k-1)/(2*n)*pi) for k in range(1, n+1)]
#Метод коллокаций
def collocation(n):
    knots = Chebyshev_roots(n)
    #knots = [-1 + 2/n*i for i in range(0, n)]
    F = [f(knot) for knot in knots]
    A = [ [Lomega(j)(knots[i-1]) for j in range(1, n+1)] for i in range(1, n+1)]
    C = np.linalg.solve(A, F)
    return A, F, C

def print_table(f, n):
    conds = []
    args = [-0.5, 0, 0.5]
    vals = [[], [], []]
    for i in range(1, n+1):
        A, F, C = f(i)
        u = build_solution(C)
        conds.append(np.linalg.cond(A, np.infty))
        for k in range(0, 3):
            vals[k].append(u(args[k]))
    table = {
        'n': [i for i in range(1, n+1)],
        'cond(A)': conds,
        'u(-0.5)': vals[0],
        'u(0)': vals[1],
        'u(0.5)': vals[2]}
    print(pd.DataFrame(data = table))


def print_info(A, F, C):
    print("Матрица системы А:")
    print(np.array(A))
    print("\n Вектор F:")
    print(np.array(F))
    print(f"\nЧисло обусловленности А: {np.linalg.cond(A, np.infty)}")
    print("Коэффициенты разложения решения: ")
    print(np.array(C))
    print("\nЗначения решения в точках -0.5, 0, 0.5:") 
    u = build_solution(C)
    args = [-0.5, 0, 0.5]
    print(np.array([u(arg) for arg in args]))

print("Метод Галеркина: ")
for i in range(1, 7):
    print(f"\nn = {i}:")
    A, F, C = Galerkin( i)
    print_info(A, F, C)


print("\n\nМетод коллокаций: ")
for i in range(1, 7):
    print(f"\nn = {i}:")
    A, F, C = collocation(i)
    print_info(A, F, C)

print("метод Галеркина")
print_table(Galerkin, 7)

print("Метод коллокаций")
print_table(collocation , 7)


_, _, C = Galerkin(7)
_, _, C_ = collocation(2)
x_plot = np.linspace(-1, 1, 100)
y_plot = np.array([build_solution(C)(x) for x in x_plot])
y_plot_ = np.array([build_solution(C_)(x) for x in x_plot])
plt.plot(x_plot, y_plot, "-", label="n=7: Метод Галеркина")
plt.plot(x_plot, y_plot_, "-.", label="n=2: Метод коллокаций")

plt.legend()
plt.show()
