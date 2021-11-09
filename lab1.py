import numpy as np

a = np.array([[-402.94, 200.02],[1200.12, -60.96]], float)
da = np.array([[0,0],[0,0]], float)
b = np.array([200,-600], float)
db = np.array([-1,-1], float)

A = np.array([[9.62483 ,1.15527,-2.97153],[1.15527,7.30891,0.69138],[-2.97153,0.69138,5.79937]], float)
B = np.array([8.71670, 5.15541, 0.27384], float)


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

#Реализация метода Жордана
def Jordan(matrix):
    n, _ = matrix.shape
    #Расширяем матрицу единичной справа
    ext = np.hstack((matrix, np.identity(n, float)))

    for k in range(0, n):
        #Выбираем номер строки для перестановки
        p = np.abs(ext[k:, k]).argmax() + k
        if p != k:
            ext[k], ext[p] = ext[p].copy(), ext[k].copy()

        tmp = ext[k, k]
        ext[k, k + 1:] /= tmp
        
        #Дополнительно зануляем елементы выше диагонали
        for i in range(0, n):
            if i != k:
                tmp = ext[i, k]
                ext[i, k + 1:] -= ext[k, k + 1:] * tmp
    #ext = (A | B), Возвращяем только B
    return ext[:, n:]


#Реализация LU-разложения
def LU_decomposition(matrix):
    n, _ = matrix.shape
    L = np.zeros_like(matrix)
    U = np.zeros_like(matrix)

    for i in range(0, n):
        for j in range(i, n):
            L[j, i] = matrix[j, i] - np.sum(L[j, :i] * U[:i, i])
            U[i, j] = (matrix[i, j] - np.sum(L[i, :i] * U[:i, j])) / L[i, i]

    return L, U
#Считаем определитель матрицы с помощью Lu-разложения
def det_LU( L, U):
    n, _ = L.shape
    res = 1
    for i in range(0, n):
        res *= L[i, i]
    return res

##Фактическая относительная погрешность
def solve_error(matrix, dmatrix, vector, dvector):
    x = np.linalg.solve(matrix, vector)
    print("reshenie ")
    print(x)
    x_ =  np.linalg.solve(matrix + dmatrix, vector + dvector)
    print("reshenie_ ")
    print(x_)
    return  np.linalg.norm(x - x_) / np.linalg.norm(x)
##Оценка для фактической относительной погрешности
def error_estimate(matrix, dmatrix, vector, dvector):    
    err_est = np.linalg.cond(matrix) / (1. - np.linalg.norm(dmatrix) / np.linalg.norm(matrix)) *\
                    (np.linalg.norm(dvector) / np.linalg.norm(vector) + np.linalg.norm(dmatrix) / np.linalg.norm(matrix))
    return err_est 


print("Число обусловленности матрицы = ", np.linalg.cond(a))
error = solve_error(a, da, b, db)
estimate = error_estimate(a, da, b, db)
print("Фактическая погрешность = ", error)
print("Оценка для фактической погрешности = ", estimate)


print("\n\nРешение методом Гаусса:")
x = Gauss(A.copy(), B.copy())
print(x)
print("Компоненты вектора невязки:")
print(B - A.dot(x))


print("Обратная матрица, найденная методом Жордана:")
A_ = Jordan(A.copy())
print(A_)

print("Проверим, что она действительно обратная: ")
e = A@A_
print(e)

print()
print("Выполним LU-разложение:")
L, U = LU_decomposition(A.copy())
print("L = ")
print(L)
print("U = ")
print(U)
print("Найдем определитель матрицы: ")
print(det_LU(L, U))



