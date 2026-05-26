import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import generate_data, StandardScaler, train_test_split_stratified
from src.perceptron import Perceptron
from src.metrics import accuracy_score

def run_basic_training():
    print("=== 1. Подготовка данных ===")
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Размер обучающей выборки: {X_train_scaled.shape}")
    print(f"Размер тестовой выборки: {X_test_scaled.shape}")

    print("\n=== 2. Обучение перцептрона ===")
    model = Perceptron(init_type='small_random')
    model.fit(
        X_train=X_train_scaled, 
        y_train=y_train, 
        X_val=X_test_scaled, 
        y_val=y_test, 
        epochs=100, 
        lr=0.1, 
        batch_size=32
    )
    print("Обучение завершено")

    print("\n=== 3. Оценка качества ===")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    
    print(f"Accuracy на обучающей выборке: {acc_train:.4f}")
    print(f"Accuracy на тестовой выборке:  {acc_test:.4f}")

    # Сохраняем в папку report/learning_plots/
    print("\n=== 4. Визуализация ===")
    plot_basic_results(model, X_test_scaled, y_test)
    

# =========================================================
# БЛОК ВИЗУАЛИЗАЦИИ 
# =========================================================

def plot_basic_results(model, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Результаты базового обучения", fontsize=16)

    # Левый график: Функция потерь 
    axes[0].plot(model.history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(model.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    axes[0].set_title("Функция потерь (Loss)")
    axes[0].set_xlabel('Эпохи')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Правый график: Разделяющая граница 
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    w, b = model.w, model.b
    if w[1] != 0:
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(w[0] * x_line + b) / w[1]
        axes[1].plot(x_line, y_line, color='black', linewidth=2, label='w^T x + b = 0')
    
    axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap='coolwarm')

    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    
    axes[1].set_title("Разделяющая граница на тесте")
    axes[1].set_xlabel('Признак 1 (Стандартизованный)')
    axes[1].set_ylabel('Признак 2 (Стандартизованный)')
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("report/learning_plots", exist_ok=True)
    
    save_path = "report/learning_plots/basic_training_results.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"График сохранен в: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    run_basic_training()