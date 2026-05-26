import numpy as np

class Perceptron:
    def __init__(self, init_type='small_random', random_state=42):
        self.init_type = init_type
        self.random_state = random_state
        self.w = None
        self.b = None
        self.v_w = None 
        self.v_b = None
        self.history = {'train_loss': [], 'val_loss': []}

    def _initialize_weights(self, n_features):
        np.random.seed(self.random_state)
        if self.init_type == 'zeros':
            self.w = np.zeros(n_features)
        elif self.init_type == 'small_random':
            self.w = np.random.randn(n_features) * 0.01
        elif self.init_type == 'large_random':
            self.w = np.random.normal(0, 10, n_features)
        else:
            raise ValueError("Неизвестный тип инициализации")
        self.b = 0.0

        # Инициализируем скорости для Momentum нулями
        self.v_w = np.zeros(n_features)
        self.v_b = 0.0

    def sigmoid(self, z):
        # Сигмоидная функция активации
        # σ(z) = 1 / (1 + e^(-z))
        z = np.clip(z, -250, 250) # np.clip предотвращает переполнение в exp
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        # Умножение матрицы входов X (размер [m x d]) на вектор весов W [d] + смещение b
        # Z = X * W + b
        z = np.dot(X, self.w) + self.b
        # y_hat = σ(Z)
        return self.sigmoid(z)

    def compute_loss(self, X, y_true, loss_type='bce', l2_coef=0.0):
        loss = 0.0
        if loss_type == 'bce':
            y_pred = self.forward(X)
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            
            # Вычисление бинарной кросс-энтропии (BCE)
            # L = -(1/m) * Σ [ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            
        elif loss_type == 'hinge':
            # Переводим метки из {0, 1} в {-1, 1}
            y_true_hinge = np.where(y_true <= 0, -1, 1)
            z = np.dot(X, self.w) + self.b

            # Вычисление Hinge Loss
            # L = max(0,1 − y*z)
            loss = np.mean(np.maximum(0, 1 - y_true_hinge * z))
            
        # Добавляем штраф L2-регуляризации
        if l2_coef > 0:
            loss += (l2_coef / 2) * np.sum(self.w ** 2)
            
        return loss

    # обучение 
    def fit(
        self, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        epochs=100, 
        lr=0.1, 
        batch_size=32,
        loss_type='bce',
        l2_coef=0.0,
        beta=0.0
    ):
        n_samples, n_features = X_train.shape
        self._initialize_weights(n_features)
        
        # Сброс истории при новом обучении
        self.history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Перемешивание перед каждой эпохой 
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Обучение по мини-батчам
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                m = X_batch.shape[0]

                # Вычисление градиентов в зависимости от loss_type

                if loss_type == 'bce':
                    # Прямой проход (предсказание модели для текущего батча)
                    y_pred = self.forward(X_batch)
                    # Ошибка предсказания 
                    # dz = y_hat - y
                    dz = y_pred - y_batch

                elif loss_type == 'hinge':
                    y_batch_hinge = np.where(y_batch <= 0, -1, 1)
                    z = np.dot(X_batch, self.w) + self.b
                    
                    # Если точка классифицирована неверно или неуверенно (y*z < 1)
                    condition = (y_batch_hinge * z) < 1
                    # Производная функции max(0, 1 - yz)
                    # Если условие выполнено, градиент равен -y. Если нет - 0
                    dz = np.where(condition, -y_batch_hinge, 0)

                # Градиент функции потерь по весам W
                # dw = (1/m) * X^T * dz
                dw = (1 / m) * np.dot(X_batch.T, dz)
                # Градиент функции потерь по смещению b
                # db = (1/m) * Σ dz
                db = (1 / m) * np.sum(dz)

                # Влияние L2-регуляризации на градиент
                if l2_coef > 0:
                    dw += l2_coef * self.w

    
                if beta > 0:
                    #  Momentum
                    # Накапливаем скорость      
                    self.v_w = beta * self.v_w + dw
                    self.v_b = beta * self.v_b + db
                    # Шагаем с учетом скорости
                    self.w -= lr * self.v_w
                    self.b -= lr * self.v_b
                else:
                    # Обычный SGD
                    # Шаг градиентного спуска
                    # w = w - η * dw (где η (эта) - это lr, скорость обучения)
                    self.w -= lr * dw
                    # b = b - η * db
                    self.b -= lr * db

            # Сохраняем loss на всю эпоху
            train_loss = self.compute_loss(X_train, y_train, loss_type, l2_coef)
            val_loss = self.compute_loss(X_val, y_val, loss_type, l2_coef)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

    def predict(self, X, threshold=0.5):
        y_pred = self.forward(X)
        return (y_pred >= threshold).astype(int)