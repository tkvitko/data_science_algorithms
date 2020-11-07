import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def mserror(X, w, y_pred):
    # реализуем функцию, определяющую среднеквадратичную ошибку
    y = X.dot(w)
    return (sum((y - y_pred) ** 2)) / len(y)


def sgd(data, l2=False, l1=False):
    # стохастичесикй грандиентный спуск

    # инициализируем начальный вектор весов
    w = np.zeros(2)
    # список векторов весов после каждой итерации
    w_list = [w.copy()]
    # список значений ошибок после каждой итерации
    errors = []
    # шаг градиентного спуска
    eta = 0.01
    # максимальное число итераций
    max_iter = 1e5
    # критерий сходимости (разница весов, при которой алгоритм останавливается)
    min_weight_dist = 1e-8
    # зададим начальную разницу весов большим числом
    weight_dist = np.inf
    # счетчик итераций
    iter_num = 0
    np.random.seed(1234)

    regularity = True
    # ход градиентного спуска
    while weight_dist > min_weight_dist and iter_num < max_iter and regularity == True:
        # генерируем случайный индекс объекта выборки
        train_ind = np.random.randint(data.shape[0])

        new_w = w - 2 * eta * np.dot(data[train_ind].T, (np.dot(data[train_ind], w) - target[train_ind])) / \
                target.shape[0]

        weight_dist = np.linalg.norm(new_w - w, ord=2)
        w_list.append(new_w.copy())

        errors.append(mserror(data, new_w, target))
        iter_num += 1
        w = new_w

        # L2-регуляризация
        if l2:
            C = 0.01    # константа, которую не должна превысить норма вектора весов
            norm = 0    # начальное значение нормы
            temp_list = new_w.tolist()  # без этого не понял, как обратиться к элементам ndarray
            for i in range(2):  # количество признаков
                norm += temp_list[i] ** 2
                if norm > C:
                    regularity = False  # если превысили С, прекращаем спуск

        # L1-регуляризация
        if l1:
            C = 0.01    # константа, которую не должна превысить норма вектора весов
            norm = 0    # начальное значение нормы
            temp_list = new_w.tolist()  # без этого не понял, как обратиться к элементам ndarray
            for i in range(2):  # количество признаков
                if temp_list[i] >= 0:   # нужен модуль, потому так
                    norm += temp_list[i]
                else:
                    norm -= temp_list[i]
                if norm > C:
                    regularity = False  # если превысили С, прекращаем спуск

    print(
        f'В случае использования стохастического градиентного спуска функционал ошибки составляет {round(errors[-1], 4)}')
    return errors


def gd(data):
    # обычный градиентный спуск

    # возьмем нулевые начальные веса
    w = np.zeros(2)
    # список векторов весов после каждой итерации
    w_list = [w.copy()]
    # список значений ошибок после каждой итерации
    errors = []
    # шаг градиентного спуска
    eta = 0.01
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
        new_w = w - 2 * eta * np.dot(data.T, (np.dot(data, w) - target)) / target.shape[0]

        weight_dist = np.linalg.norm(new_w - w, ord=2)
        w_list.append(new_w.copy())
        errors.append(mserror(data, new_w, target))
        iter_num += 1
        w = new_w

    w_list = np.array(w_list)

    print(f'В случае использования градиентного спуска функционал ошибки составляет {round(errors[-1], 4)}')
    return errors


if __name__ == '__main__':

    # сгенерируем набор данных
    my_data, target, coef = datasets.make_regression(n_samples=1000, n_features=2, n_informative=2, n_targets=1,
                                                     noise=5, coef=True, random_state=2)

    # масштабирование признаков
    # Получим средние значения и стандартное отклонение по столбцам

    means = np.mean(my_data, axis=0)
    stds = np.std(my_data, axis=0)
    # параметр axis указывается для вычисления значений по столбцам, а не по всему массиву
    # (см. документацию в разделе источников)

    # вычтем каждое значение признака из среднего и поделим на стандартное отклонение
    for i in range(my_data.shape[0]):
        for j in range(my_data.shape[1]):
            my_data[i][j] = (my_data[i][j] - means[j]) / stds[j]

    print('Стохастический ГС')
    my_errors_sgd = sgd(my_data)
    print('Обычный ГС')
    my_errors_gd = gd(my_data)
    print('Стохастический ГС с L2-регуляризацией')
    my_errors_sgd_l2 = sgd(my_data, l2=True)
    print('Стохастический ГС с L1-регуляризацией')
    my_errors_sgd_l1 = sgd(my_data, l1=True)

    # Визуализируем изменение функционала ошибки
    plt.plot(range(len(my_errors_sgd)), my_errors_sgd)
    plt.plot(range(len(my_errors_gd)), my_errors_gd)
    plt.plot(range(len(my_errors_sgd_l1)), my_errors_sgd_l1)
    plt.plot(range(len(my_errors_sgd_l2)), my_errors_sgd_l2)
    plt.title('MSE')
    plt.xlabel('Iteration number')
    plt.ylabel('MSE')
    plt.show()

    # На графике функционал ошибки обычного градиентного спуска снижается гораздо быстрее, чем стохастического.
    # Как я понимаю, это из-за того, что мы на каждой итерации оперируем градиентом всей матрицы.
    # После добавления L2 и L1 регуляризаций новых графиков не увидел. Если я правильно понял, графики идут так же,
    # как первый, но заканчиваются раньше.
