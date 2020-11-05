import numpy as np
import matplotlib.pyplot as plt


def mserror(X, w, y_pred):
    # реализуем функцию, определяющую среднеквадратичную ошибку
    y = X.dot(w)
    return (sum((y - y_pred) ** 2)) / len(y)


def gradient_descent(eta):

    np.random.seed(1234)

    # Возьмем 2 признака и 1000 объектов
    n_features = 2
    n_objects = 1000

    # сгенерируем вектор истинных весов
    w_true = np.random.normal(size=(n_features,))

    # сгенерируем матрицу X, вычислим Y с добавлением случайного шума
    X = np.random.uniform(-7, 7, (n_objects, n_features))
    Y = X.dot(w_true) + np.random.normal(0, 0.5, size=(n_objects))

    # возьмем нулевые начальные веса
    w = np.zeros(n_features)

    # список векторов весов после каждой итерации
    w_list = [w.copy()]

    # список значений ошибок после каждой итерации
    errors = []

    # шаг градиентного спуска

    # максимальное число итераций
    max_iter = 1e4

    # критерий сходимости (разница весов, при которой алгоритм останавливается)
    min_weight_dist = 1e-8

    # зададим начальную разницу весов большим числом
    weight_dist = np.inf

    # счетчик итераций
    iter_num = 0

    # ход градиентного спуска
    while weight_dist > min_weight_dist and iter_num < max_iter:
        new_w = w - 2 * eta * np.dot(X.T, (np.dot(X, w) - Y)) / Y.shape[0]
        weight_dist = np.linalg.norm(new_w - w, ord=2)

        w_list.append(new_w.copy())
        errors.append(mserror(X, new_w, Y))

        iter_num += 1
        w = new_w

    w_list = np.array(w_list)

    #print(f'В случае использования градиентного спуска функционал ошибки составляет {round(errors[-1], 4)}')

    # Визуализируем изменение весов (красной точкой обозначены истинные веса, сгенерированные вначале)
    # plt.figure(figsize=(13, 6))
    # plt.title('Gradient descent')
    # plt.xlabel(r'$w_1$')
    # plt.ylabel(r'$w_2$')
    #
    # plt.scatter(w_list[:, 0], w_list[:, 1])
    # plt.scatter(w_true[0], w_true[1], c='r')
    # plt.plot(w_list[:, 0], w_list[:, 1])
    #
    # plt.show()

    return iter_num

if __name__ == '__main__':

    etas = [0.0001, 0.001, 0.01, 0.1, 1]

    for eta in etas:
        iter_num = gradient_descent(eta)
        print(f'При eta={eta} мы имеем {iter_num} итераций')