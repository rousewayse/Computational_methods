#!/bin/env python3
import numpy as np
import pandas as pd
from math import ceil

def alph1(t):
    return 0

def alph2(t):
    return -1

def betta1(t):
    return 1

def betta2(t):
    return 0

def alph(t):
    return 0

def betta(t):
    return u(1, t)

def phi(x: float):
    return u(x, 0)

def p(x: float):
    return x + 1
def maxp():
    return p(D[0][1])
def b(x, t):
    return 0

def c(x, t):
    return 0

def u(x, t):
    return (x ** 3) + (t ** 3)

def f(x, t):
    return 3 * (t ** 2) - 9 * (x ** 2) - 6 * x


#Строит таблицу со значениями точного решения
def build_acc_vals(N, M, h, tau, D):
    x = lambda i: D[0][0] + h*i
    t = lambda i: D[1][0] + tau*i

    return [[u(x(i), t(k)) for i in range(0, N+1)] for k in range(0, M+1)]
#i from 1 to N-1
#k from 1 to M
#Вычисляет значение разностного оператора, который приближает дифференциальный
def Lu(h, i, k, u_vals, x, t):
    return p((i+0.5)*h)*(u_vals[k][i+1] - u_vals[k][i])/(h**2) - p((i-0.5)*h)*(u_vals[k][i] - u_vals[k][i-1])/(h**2) + b(x(i), t(k))*(u_vals[k][i+1] - u_vals[k][i-1])/(2*h) + c(x(i), t(k))*u_vals[k][i];


#Явная разностная схема
# in (xi, tk-1)
def expl_sub(N, M, h, tau, D):
        x = lambda i: D[0][0] + h*i
        t = lambda i: D[1][0] + tau*i

        u_vals = []
        def find_u0i():
            return [phi(x(i)) for i in range(0, N+1)]
        # i from 1 to N-1
        def find_uki(k):
            return [u_vals[k-1][i] + tau*(Lu(h, i, k-1, u_vals, x, t) + f(x(i), t(k-1))) for i in range(1, N)]
        def find_uk0(k):
            return (alph(t(k)) + alph2(t(k))*(4*u_vals[k][1] - u_vals[k][2])/(2*h))/(alph1(t(k)) + (3*alph2(t(k))/(2*h)))
        def find_ukN(k):
            return (betta(t(k)) - betta2(t(k))*(-4*u_vals[k][N-1] + u_vals[k][N-2])/(2*h))/(betta1(t(k)) + betta2(t(k))*3/(2*h))
        
        u_vals.append(find_u0i())
        #вычисляем значения на каждом слое
        for k in range(1, M+1):
            u_vals.append([0] + find_uki(k) + [0])
            u_vals[k][0] = find_uk0(k)
            u_vals[k][N] = find_ukN(k)

        return u_vals

#Неявная схема
def impl_schma(sgm, N, M, h, tau, D):
    x = lambda i: D[0][0] + h*i
    t = lambda i: D[1][0] + tau*i

    u_vals = []
    #Значения решения на нулевом слое
    def find_u0i():
        return [phi(x(i)) for i in range(0, N+1)]

    def t_(k):
        if sgm == 0.5: return t(k) - tau/2
        if sgm == 0: return t(k-1)
        return t(k)
#Для вычисления значений решения на каждом слое используем метод прогонки
    def find_uki(k):
        Ak = [0] + [sgm*(p(x(i-0.5))/(h**2) - b(x(i),t(k))/(2*h)) for i in range(1, N)] + [-betta2(t(k))/h ]
        Bk = [-alph1(t(k)) - alph2(t(k))/h] + [sgm*(p(x(i+0.5))/(h**2) + p(x(i-0.5))/(h**2) - c(x(i), t(k))) + 1/tau for i in range(1, N)] + [ -betta1(t(k)) - betta2(t(k))/h]
        Ck = [-alph2(t(k))/h] + [sgm*(p(x(i+0.5))/(h**2) + b(x(i), t(k))/(2*h)) for i in range(1, N)] + [0] 
        Gk = [alph(t(k))] +[-1/tau*u_vals[k-1][i] - (1-sgm)*Lu(h,i,k-1, u_vals, x, t) - f(x(i), t_(k)) for i in range(1, N)]+ [betta(t(k))]
        #ui = si*ui+1 +ti
        def sk_tk():
            sk = [Ck[0]/Bk[0]]
            tk = [-Gk[0]/Bk[0]]
            for i in range(1, N+1):
                sk.append(Ck[i]/(Bk[i] - Ak[i]*sk[-1]))
                tk.append((Ak[i]*tk[-1] - Gk[i])/(Bk[i] - Ak[i]*sk[-1]))
            return sk, tk
        sk, tk = sk_tk()
        
        #Создаем таблицу значений решения
        uk_vals = [tk[N]] 
        for i in range (N-1, -1, -1):
            uk_vals.append(sk[i]*uk_vals[-1] + tk[i])
        return list(reversed(uk_vals)) 
    
    #Вычисляем значения решеня на каждом слое
    u_vals.append(find_u0i())
    for k in range(1, M+1):
        u_vals.append(find_uki(k))
    return u_vals


#Возвращает тау из условий устройчивости
#returns (tau, M)
def get_stable_tau(N, h, D):
    return  (D[1][1] -  D[1][0])/ceil(0.2*maxp()/(h**2)),int(ceil(0.2*maxp()/(h**2)))

#находит норму матрицы разности решений
def snorm_diff(sol1, sol2, N, M):
    temp = []
    for k in range(0, M+1):
        temp.append(max(list(map(lambda a, b: abs(a-b), sol1[k], sol2[k]))))
    return max(temp)

def print_expl_acc_check_table(D):
    print("Таблица точности для явной схемы:")
    #h tau ||... || ||...||
    t_data = [[], [], [], []]
    M = 10
    tau =  (D[1][1] -  D[1][0])/M 
    h_set = [0.2, 0.1, 0.05]
    N_set = list(map(lambda i: int((D[0][1] -  D[0][0])/i), h_set))
    for i in range(0, len(h_set)):
        
        tau, M = get_stable_tau(N_set[i], h_set[i], D)
        t_data[0].append(h_set[i])
        t_data[1].append(tau)
        t_data[2].append(snorm_diff(build_acc_vals(N_set[i], M, h_set[i], tau, D), expl_sub(N_set[i], M, h_set[i], tau, D), N, M))
        t_data[3] = None
    table = {
        'h': t_data[0],
        'tau': t_data[1],
        '||J_ex - u(h, tau)||': t_data[2],
        '||u(h,tau) - u(2h, tau\')||':  t_data[3]
    }
    print(pd.DataFrame(data = table))


def print_impl_schma_acc_table(D):
    print("Таблицы точности для неявной схемы:")
    M = 10
    tau = (D[1][1] -  D[1][0])/M
    sigma_set = [1, 0.5, 0]
    h_set = [0.2, 0.1, 0.05]
    N_set = list(map(lambda i: int((D[0][1] -  D[0][0])/i), h_set))
    for sgm in sigma_set:
        t_data = [ [], [], [], []]
        for i in range(0, len(h_set)):
            t_data[0].append(h_set[i])
            t_data[1].append(tau)
            t_data[2].append(snorm_diff(build_acc_vals(N_set[i], M, h_set[i], tau, D), impl_schma(sgm, N_set[i], M, h_set[i], tau, D), N, M))
            t_data[3] = None
            table = {
                'h': t_data[0],
                'tau': t_data[1],
                '||J_ex - u(h, tau)||': t_data[2],
                '||u(h,tau) - u(2h, tau\')||':  t_data[3]
        }
        print(f"\n\nsigma = {sgm}")
        print(pd.DataFrame(data = table))

#Сетка решения через явный метод
def print_expl_large_net(N, M, D):
    
    print("\n\nКрупная сетка для явной схемы:")
    tau = (D[1][1] -  D[1][0])/M
    h = (D[0][1] -  D[0][0])/N
    t_data = expl_sub(N, M, h, tau, D)
    print(pd.DataFrame(t_data, index = [ f"{k*tau}" for k in range(0, M+1)], columns = [f"{i*h}" for i in range(0, N+1)]))
#Сетка решения через неявный метод
def print_impl_schma_large_net(N, M, D, sgm):
    tau = (D[1][1] -  D[1][0])/M
    h = (D[0][1] -  D[0][0])/N
    t_data = impl_schma(sgm, N, M, h, tau, D)
    
    print("\n\nКрупная сетка для неявной схемы:")
    print(f"sigma = {sgm}")
    print(pd.DataFrame(t_data, index = [ f"{k*tau}" for k in range(0, M+1)], columns = [f"{i*h}" for i in range(0, N+1)]))

def print_acc_sol_large_net(N, M, D):
    print("Точное решение на крупной сетке:")
    tau = (D[1][1] -  D[1][0])/M
    h = (D[0][1] -  D[0][0])/N
    t_data = build_acc_vals(N, M, h, tau, D)
    print(pd.DataFrame(t_data, index = [ f"{k*tau}" for k in range(0, M+1)], columns = [f"{i*h}" for i in range(0, N+1)]))
#Входные данные
D = ((0.0, 1.0), (0.0, 0.1)) 
N = 5
M = 10
h = (D[0][1] -  D[0][0])/N
tau = (D[1][1] -  D[1][0])/M
sigms = [0, 0.5, 1]
#x = lambda i: D[0][0] + h*i
#t = lambda i: D[1][0] + tau*i


print_expl_acc_check_table(D)
print_impl_schma_acc_table(D)

print_acc_sol_large_net(N, M, D)
print_expl_large_net(N,M,D)
print_impl_schma_large_net(N, M, D, sigms[0])
print_impl_schma_large_net(N, M, D, sigms[1])
print_impl_schma_large_net(N, M, D, sigms[2])
