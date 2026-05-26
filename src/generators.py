import numpy as np

def add_noise(y, noise_prob, random_state=42):
    """
    Инвертирует метки класса с вероятностью noise_prob (добавление шума)
    """
    if noise_prob > 0:
        np.random.seed(random_state)
        # Создаем маску: True с вероятностью noise_prob
        flip_mask = np.random.rand(len(y)) < noise_prob
        # Инвертируем метки (1 станет 0, а 0 станет 1)
        y[flip_mask] = 1 - y[flip_mask]
    return y

def generate_gaussian_clouds(n_samples=500, noise_prob=0.0, random_state=42):
    """
    Генерация двух линейно разделимых гауссовых облаков
    """
    np.random.seed(random_state)
    n_class = n_samples // 2

    # Класс 0 (центр внизу слева)
    center0 = [-2, -2]
    cov0 = [[1, 0], [0, 1]] # Единичная матрица (круглые облака)
    X0 = np.random.multivariate_normal(center0, cov0, n_class)
    y0 = np.zeros(n_class)

    # Класс 1 (центр вверху справа)
    center1 = [2, 2]
    cov1 = [[1, 0], [0, 1]]
    X1 = np.random.multivariate_normal(center1, cov1, n_samples - n_class)
    y1 = np.ones(n_samples - n_class)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    return X, add_noise(y, noise_prob, random_state)

def generate_xor(n_samples=500, noise_prob=0.0, random_state=42):
    """
    Генерация данных типа XOR (по углам квадрата)
    """
    np.random.seed(random_state)
    n_per_cluster = n_samples // 4
    
    # Центры 4-х облаков (по углам)
    centers = [[2, 2], [-2, -2], [-2, 2], [2, -2]]
    # Метки: первая диагональ - 0, вторая - 1
    y_labels = [0, 0, 1, 1]
    
    X, y = [], []
    for center, label in zip(centers, y_labels):
        # Делаем облака плотными (ковариация 0.5)
        X.append(np.random.multivariate_normal(center, [[0.5, 0], [0, 0.5]], n_per_cluster))
        y.append(np.full(n_per_cluster, label))
        
    X = np.vstack(X)
    y = np.hstack(y)
    return X, add_noise(y, noise_prob, random_state)

def generate_circles(n_samples=500, noise_prob=0.0, random_state=42):
    """
    Генерация вложенных окружностей (внутренний круг и внешнее кольцо)
    """
    np.random.seed(random_state)
    n_class = n_samples // 2

    # Класс 0 (внутренний круг, радиус от 0 до 1.5)
    radius_inner = np.random.uniform(0, 1.5, n_class)
    angle_inner = np.random.uniform(0, 2 * np.pi, n_class)
    X0 = np.c_[radius_inner * np.cos(angle_inner), radius_inner * np.sin(angle_inner)]
    y0 = np.zeros(n_class)

    # Класс 1 (внешнее кольцо, радиус от 2.5 до 4.0)
    radius_outer = np.random.uniform(2.5, 4.0, n_samples - n_class)
    angle_outer = np.random.uniform(0, 2 * np.pi, n_samples - n_class)
    X1 = np.c_[radius_outer * np.cos(angle_outer), radius_outer * np.sin(angle_outer)]
    y1 = np.ones(n_samples - n_class)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    return X, add_noise(y, noise_prob, random_state)