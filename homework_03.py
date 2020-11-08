import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def calc_logloss(y, y_pred):
    # Задание 1
    # Непопадание нулей в np.log:
    # for i in y_pred.size:
    #     if y_pred[i] == 0 or y_pred[i] == 1:
    #         y.pop[i]
    #         y_pred.pop[i]
    # Как правильно вынимать элементы ndarray, не придумал, потому код не работает.

    err = - np.mean(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred))
    return err


def calc_std_feat(x):
    res = (x - x.mean()) / x.std()
    return res


def sigmoid(z):
    res = 1 / (1 + np.exp(-z))
    return res


def eval_model(X, y, iterations, alpha=1e-4):
    np.random.seed(42)
    W = np.random.randn(X.shape[0])
    n = X.shape[1]
    for i in range(1, iterations + 1):
        z = np.dot(W, X)
        y_pred = sigmoid(z)
        err = calc_logloss(y, y_pred)
        W -= alpha * (1 / n * np.dot((y_pred - y), X.T))
    if i % (iterations / 10) == 0:
        print(i, W, err)
    return W


def calc_pred_proba(W, X):
    # Задание 3
    # Возвращает предсказанную вероятность класса 1
    y_pred_proba = 1 / (1 + np.exp(-np.dot(W, X)))
    # взял формулу из методички
    # почему-то вероятность всегда получается равной 1 :(
    # Вообще, как я понимаю, она долна равняться y_pred.
    return y_pred_proba


def calc_pred(W, X):
    # Задание 4
    # Возвращает предсказанный класс
    # Сильно не уверен, что так правильно делать, но идей лучше нет :(
    y_pred_proba = calc_pred_proba(W, X)
    classes = []
    for prob in y_pred_proba:
        if prob >= 0.5:
            class_ = '+1'
        else:
            class_ = '-1'
        classes.append(class_)

    return classes


def acceracy():
    pass


if __name__ == '__main__':
    X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 2, 1, 3, 0, 5, 10, 1, 2],
                  [500, 700, 750, 600, 1450,
                   800, 1500, 2000, 450, 1000],
                  [1, 1, 2, 1, 2,
                   1, 3, 3, 1, 2]], dtype=np.float64)

    y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.float64)

    X_st = X.copy()
    X_st[2, :] = calc_std_feat(X[2, :])
    # Задание 2
    # err=0.59 при 100 000 итераций и шаге e-5 (падает при увеличении количества итераций).
    # если увеличивать шаг, err становится еще меньше. Но не уверен, что так делать правильно.
    W = eval_model(X_st, y, iterations=100000, alpha=1e-5)

    print(calc_pred_proba(W, X))
    print(calc_pred(W, X))
