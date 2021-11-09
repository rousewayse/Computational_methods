#!/bin/env python3
import numpy as np
import pandas as pd

A = np.array([[-1.47887, -0.09357, 0.91259], [-0.09357, 1.10664, 0.03298], [0.91259, 0.03298, -1.48225]], float)
#Y0 = np.ones((A.shape[0], 1), float)
Y0 = np.array([[-35],[ 55], [53]],float)
eps = 0.001

#Апостериорная оценка погрешности
def aposterior_err(A, eig_val, eig_vec):
    temp = np.linalg.norm(A.dot(eig_vec) - eig_val*eig_vec, 2) / np.linalg.norm(eig_vec, 2)
    return temp
#Степенной метод
def power(A, Y0, eps):
    iters = 0
    temp = np.array([abs(i) for i in Y0], float).argmax()
    eig_vecs = [Y0.copy()/Y0[temp]]
    eig_vals = [ A.dot(eig_vecs[iters])[temp][0]] 
    
    while aposterior_err(A, eig_vals[iters], eig_vecs[iters]) > eps:
        
        eig_vecs.append(A.dot(eig_vecs[iters]))
        ind = np.array([abs(i) for i in eig_vecs[-1]], float ).argmax()
        eig_vecs[-1] /= eig_vecs[-1][ind] 
        eig_vals.append(A.dot(eig_vecs[-1])[ind][0])

        iters += 1
    return eig_vals, eig_vecs
#Метод скалярных произведений
def scalar(A, Y0, eps):
    iters = 0;
    temp = np.array([abs(i) for i in Y0], float).argmax()
    eig_vecs = [Y0.copy()/Y0[temp]]
    eig_vals = [A.dot(eig_vecs[-1]).dot(eig_vecs[-1].transpose())[0][0]/eig_vecs[-1].dot(eig_vecs[-1].transpose())[0][0]]
    
    while aposterior_err(A, eig_vals[-1], eig_vecs[-1]) > eps:
        eig_vecs.append(A.dot(eig_vecs[-1]))
        ind = np.array([abs(i) for i in eig_vecs[-1]], float ).argmax()
        eig_vecs[-1] /= eig_vecs[-1][ind]
        eig_vals.append(A.dot(eig_vecs[-1]).dot(eig_vecs[-1].transpose())[0][0]/eig_vecs[-1].dot(eig_vecs[-1].transpose())[0][0])
    
    return eig_vals, eig_vecs

def print_table(eig_vals, eig_vecs, A):
    acc_val = np.array([abs(i) for i in np.linalg.eig(A)[0]]).argmax()
    acc_val = np.linalg.eig(A)[0][acc_val]
    table = {
        'k': [i for i in range(0, len(eig_vals))],
        'eig_val[k]': eig_vals,
        'eig_val[k] - eig_val[k-1]': [0] + [eig_vals[i] - eig_vals[i-1] for i in range (1, len(eig_vals))],
        'eig_val[k] - acc_eig_val': [ i - acc_val for i in eig_vals],
        '||vA*eig_vec[k] - eig_val[k]*eig_vec[k]||': [np.linalg.norm(A.dot(eig_vecs[i]) - eig_vals[i]*eig_vecs[i], 2) for i in range(0, len(eig_vals))],
        'Error_estimate': [ aposterior_err(A, eig_vals[i], eig_vecs[i]) for i in range(0, len(eig_vals))]
    }
    print(pd.DataFrame(data = table))
#'||A*eig_vec[k] - eig_val[k]*eig_vec[k]||': [np.linalg.norm(A.dot(eig_vecs[i]) - eig_vals[i]*eig_vecs[i], 2) for i in range(0, len(eig_vals))],


print("С.ч, найденное библиотечным методом:")
print(np.linalg.eig(A)[0][0])
eig_vals, eig_vecs = power(A, Y0, eps)
print("С.ч и с.в, найденные с помощью степенного метода:")
print(eig_vals[-1])
print((eig_vecs[-1][0][0], eig_vecs[-1][1][0], eig_vecs[-1][2][0]))

eig_vals, eig_vecs  = scalar(A, Y0, eps)
print("С.ч и с.в, найденные с помощью метода скалярных произведений:")
print(eig_vals[-1])
print((eig_vecs[-1][0][0], eig_vecs[-1][1][0], eig_vecs[-1][2][0]))

eig_vals, eig_vecs = power(A, Y0, eps)
print("Таблица степенного метода: ")
print_table(eig_vals, eig_vecs, A)

eig_vals, eig_vecs  = scalar(A, Y0, eps)
print("Таблица метода скалярных произведений: ")
print_table(eig_vals, eig_vecs, A)
