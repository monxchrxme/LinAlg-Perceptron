import numpy as np
from sklearn.datasets import make_classification

def generate_data(n_samples=500, random_state=42):
    """
    Генерация данных
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=1
    )
    return X, y

class StandardScaler:
    """
    Z-нормализация признаков
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        # μ (Мю) - вычисление среднего арифметического по каждому признаку
        # μ = (1/n) * Σ x_i
        self.mean = np.mean(X, axis=0)

        # σ (Сигма) - вычисление стандартного отклонения
        # σ = sqrt( (1/n) * Σ (x_i - μ)^2 )
        self.std = np.std(X, axis=0)

        # Защита от деления на 0, если признак константный
        self.std[self.std == 0] = 1e-8 
        return self

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Сначала необходимо вызвать метод fit")
        
        # преобразование Z-score
        # x' = (x - μ) / σ
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def train_test_split_stratified(X, y, test_size=0.3, random_state=42):
    """
    Разбиение данных с сохранением пропорций классов (стратификация)
    """
    np.random.seed(random_state)
    
    classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        
        n_test = int(len(cls_indices) * test_size)
        
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
        
    # Перемешиваем итоговые индексы, чтобы классы не шли подряд
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]