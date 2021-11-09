#!/bin/env python3
import numpy as np
import pandas as pd
from math import *

#Определяем пределы интегрирования [a, b]
section = (0, 1) # [a, b] = [0, 1]

#Точность для поиска решений уравнения
eps = 10**(-10)

#Точностю для поиска узлов КФ (корней многочлена Лежандра)
root_eps = 10**(-15)
#Кол-во узлов КФ Гаусса
n = 2

#Определим функции H(x, y) и f(x) из условий задания
# u(x) - integral(H(x,y)u(y)dy) = f(x)
def H(arg1, arg2):
    return 0.5*exp((arg1-0.5)*(arg2**2))

def f(arg1):
    return arg1+0.5
#def H(x, y):
#    return tanh(x * y) / 2.
#
#def f(x):
#    return x - 0.5

def quadrature_eval(coefs, knots, func):
    return  np.sum([coefs[i]*func(knots[i]) for i in range(0,len(knots))])

#Вычисление значения многочлена Лежандра
#возвращается список {P0(x), ... , Pn(x)}
def legp_eval(arg, n):
    lst = [1.0, float(arg)]
    for i in range(1, n+1):
        t = lst[i]*(2.0*float(i) + 1.0)/(float(i)+1.0)*arg - lst[i-1]*float(i)/float(i+1)
        lst.append(t)
    return lst

#Производная многочлена Лежандра 
def legp_der(arg, n):
    lst = legp_eval(arg, n)
    return n/(1.0-arg**2)*(lst[n-1] - arg*lst[n])
#Нахождение корней многочлена Лежандра
def legp_roots(n, eps):
    roots = [cos(pi*(4*i -1.0)/(4*n+2)) for i in range(1, n+1)]
    for i in range(0, n):
        temp = roots[i] - legp_eval(roots[i], n)[n]/legp_der(roots[i], n)
        while abs(temp - roots[i]) > eps:
            #print(f"{i } == {abs(temp - roots[i])}")
            roots[i] = temp
            temp = roots[i] - legp_eval(roots[i], n)[n]/legp_der(roots[i], n)
        roots[i] = temp
        #print(f"{i } == {abs(temp - roots[i])}")
    return roots


#Функция подготавливающая узлы и коэффициенты КФ Гаусса
def Gauss_prepare(n,  sect, eps):
    knots = legp_roots(n, eps)
    coefs =  [2/(1-k**2)/(legp_der(k, n)**2) for k in knots]
    knots = list(map(lambda k: (sect[1] - sect[0])/2*k + (sect[1] + sect[0])/2, knots))
    return coefs, knots


def kron(k, j):
    if k == j:
        return 1
    return 0


def mech_quadrature(f, H, n, sect, eps, root_eps):
    def sol(arg):
        return np.sum([coefs[i]*H(arg, knots[i])*z[i] for i in range(0, n)]) + f(arg)
    coefs, knots = Gauss_prepare(n,sect, root_eps)
    D = np.array([[ kron(k, j) - coefs[k]*H(knots[j], knots[k]) for k in range(0, n)] for j in range(0, n)])
    g =np.array([f(k) for k in knots])
    z = np.linalg.solve(D, g)
    def idn():
        print(f"\n\nКоличество узлов КФ: {n}")
        print("\nУзлы:")
        for i in knots:
            print(i)
        print("\nКоэффициенты КФ: ")
        for i in coefs:
            print(i)
        if (n < 5):
            print("\nМатрица системы: ")
            print(D)
            print("\nВектор правых частей: ")
            print(g)
        print(f"\nРешение системы: {z}")
        args = [sect[0], (sect[1]+sect[0])/2, sect[1]]
        print("\nЗначения решения в a, (a + b)/2, b:")
        print([sol(i)  for i in args])
        print("\n\n")
    idn()
    return sol

def find_eps_near_sol(n, sect, eps, root_eps):
        
    def sols_diff(sols, sect):
        args = [sect[0], (sect[1]+sect[0])/2, sect[1]]
        return  np.array([abs(sols[-1](i) - sols[-2](i)) for i in args], float).max()
        

    sols = [mech_quadrature(f, H, n, sect, eps, root_eps), mech_quadrature(f, H, n+1, sect, eps, root_eps)]
    iters = 2
    while sols_diff(sols, sect)>eps:
        sols.append(mech_quadrature(f, H, n+iters,sect,  eps, root_eps))
        iters+=1

    return sols

def print_table(sols, sect, n):
    a = sect[0]
    mid = (sect[0]+ sect[1])/2
    b = sect[1]

    diffs = [0.0]
    for k in range(1, len(sols)):
        tmp = np.array([abs(sols[k](i) - sols[k-1](i)) for i in [a, mid, b]], float).max()
        diffs.append(tmp)
    table = {
        'x': [f"u^{n}+{i}" for i in range(0, len(sols))],
        'a': [sol(a) for sol in sols], 
        '(a+b)/2': [sol(mid) for sol in sols],
        'b': [sol(b) for sol in sols], 
        'max | u^{i}(xi) - u^{i-1}(xi)|': diffs    
    }
 
    print(pd.DataFrame(data = table))
    a_ = sols[-1](a) - sols[-2](a)
    mid_ = sols[-1](mid) - sols[-2](mid)
    b_ = sols[-1](b) - sols[-2](b)
    print (f"u^n+{len(sols)-1} - u^n+{len(sols)-2}"  + f" =  {(a_, mid_, b_)}")    

def leg_acc(num , n = 6):
    res = []
    for i in range(0, num):
        arg = -1 + (1 -(-1))/num*i
        accs = [1, arg, (3*arg**2 - 1)/2,\
        (5*arg**3 - 3*arg)/2,\
        (35*arg**4 - 30*arg**2 +3)/8,\
        (63*arg**5 - 70*arg**3 +15*arg)/8,\
        (231*arg**6 - 315*arg**4 + 105*arg**2 -5)/16]
        appr = legp_eval(arg, n)
        res.append( list(map(lambda a, b: abs(a - b), accs, appr)))
    print(np.array([ np.array([k  for k in i], float).max() for i in res], float).max())

def leg_root_acc():
    reps  = 10**(-100)
    res = []
    for i in range(2, 7):
        temp = []
        for j in legp_roots(i, reps):
            arg = j
            accs = [1, arg, (3*arg**2 - 1)/2,\
            (5*arg**3 - 3*arg)/2,\
            (35*arg**4 - 30*arg**2 +3)/8,\
            (63*arg**5 - 70*arg**3 +15*arg)/8,\
            (231*arg**6 - 315*arg**4 + 105*arg**2 -5)/16]
            temp.append(abs(0 - accs[i]))
        res.append(temp)
    #print(res[2:4])
    res = [np.array([i for i in k], float).max() for k in res]
    
    print(res)
#legp_roots(5, 10**(-15))
sols = find_eps_near_sol(n, section, eps, root_eps)
print_table(sols, section, n)

