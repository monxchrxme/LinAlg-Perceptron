import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import generate_data, StandardScaler
from src.perceptron import Perceptron
from src.metrics import accuracy_score

def k_fold_split(n_samples, k=5, random_state=42):
    """
    Генератор индексов для K-Fold кросс-валидации
    """
    np.random.seed(random_state)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Определяем размеры каждого фолда (учитывая, что нацело может не делиться)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1 

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        # Обучающая выборка - это всё, кроме текущего фолда
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, val_indices))
        current = stop
    return folds

def run_cross_validation():
    print("=== Запуск 5-кратной кросс-валидации ===")
    X, y = generate_data(n_samples=500)

    # Сетка параметров (Grid Search)
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1]
    batch_sizes = [8, 16, 32, 64, 128]
    k_folds = 5

    folds = k_fold_split(len(X), k=k_folds)

    results = []
    best_acc = 0
    best_params = None

    print(f"{'LR':<8} | {'Batch':<8} | {'Mean Acc':<10} | {'Std Dev':<10}")
    print("-" * 45)

    # product перебирает все возможные комбинации 
    for lr, bs in product(learning_rates, batch_sizes):
        fold_accuracies = []

        for train_idx, val_idx in folds:
            X_train_f, y_train_f = X[train_idx], y[train_idx]
            X_val_f, y_val_f = X[val_idx], y[val_idx]

            # Стандартизация внутри фолда (чтобы не было утечки данных)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_f)
            X_val_scaled = scaler.transform(X_val_f)

            # Обучаем модель на текущем разбиении
            model = Perceptron(init_type='small_random')
            model.fit(
                X_train_scaled, 
                y_train_f, 
                X_val_scaled, 
                y_val_f, 
                epochs=50, 
                lr=lr, 
                batch_size=bs, 
                loss_type='bce'
            )

            # Оцениваем на валидационном фолде
            y_pred = model.predict(X_val_scaled)
            fold_accuracies.append(accuracy_score(y_val_f, y_pred))

        # Вычисляем среднее и стандартное отклонение по 5 фолдам
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)

        results.append({'lr': lr, 'batch_size': bs, 'mean': mean_acc, 'std': std_acc})
        print(f"{lr:<8} | {bs:<8} | {mean_acc:<10.4f} | {std_acc:<10.4f}")

        # Запоминаем лучшие параметры
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = {'lr': lr, 'batch_size': bs}

    print("\n=== Лучшие гиперпараметры ===")
    print(f"LR: {best_params['lr']}, Batch Size: {best_params['batch_size']} (Средняя точность: {best_acc:.4f})")

    print("\nОбучение финальной модели на всех обучающих данных...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 
    
    final_model = Perceptron(init_type='small_random')
    final_model.fit(X_scaled, y, X_scaled, y, 
                    epochs=100, lr=best_params['lr'], batch_size=best_params['batch_size'])
    print(f"Финальная модель успешно обучена! Финальный Loss: {final_model.history['train_loss'][-1]:.4f}")

    plot_cv_results(results, learning_rates, batch_sizes)

# =========================================================
# БЛОК ВИЗУАЛИЗАЦИИ 
# =========================================================

def plot_cv_results(results, learning_rates, batch_sizes):
    acc_matrix = np.zeros((len(learning_rates), len(batch_sizes)))

    for res in results:
        i = learning_rates.index(res['lr'])
        j = batch_sizes.index(res['batch_size'])
        acc_matrix[i, j] = res['mean']

    fig, ax = plt.subplots(figsize=(8, 6))
    # Рисуем тепловую карту
    cax = ax.matshow(acc_matrix, cmap='coolwarm')
    fig.colorbar(cax, label='Mean Accuracy')

    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_yticks(np.arange(len(learning_rates)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticklabels(learning_rates)
    ax.set_xlabel('Batch Size (Размер батча)', fontsize=12)
    ax.set_ylabel('Learning Rate (Скорость обучения)', fontsize=12)
    ax.set_title('Grid Search: Средняя точность (5-Fold CV)', pad=20, fontsize=14)

    for i in range(len(learning_rates)):
        for j in range(len(batch_sizes)):
            text_color = "black" if acc_matrix[i, j] > np.mean(acc_matrix) else "white"
            ax.text(j, i, f"{acc_matrix[i, j]:.4f}", ha="center", va="center", color=text_color, fontweight='bold')

    plt.tight_layout()
    os.makedirs("report/experiment_plots", exist_ok=True)
    plt.savefig("report/experiment_plots/cross_validation_exp.jpg", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_cross_validation()