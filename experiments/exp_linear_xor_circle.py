import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.generators import generate_gaussian_clouds, generate_xor, generate_circles
from src.data import StandardScaler, train_test_split_stratified
from src.perceptron import Perceptron
from src.metrics import accuracy_score

def evaluate_dataset(X, y, title):
    # Предобработка
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение
    model = Perceptron(init_type='small_random')
    model.fit(
        X_train_scaled, 
        y_train, 
        X_test_scaled, 
        y_test, 
        epochs=100, 
        lr=0.1, 
        batch_size=32
    )
    
    acc_test = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"[{title}] Test Accuracy: {acc_test:.4f}")
    
    return model, X_test_scaled, y_test

def run_generators_experiment():
    print("=== Оценка на различных датасетах (Шум 5%) ===")
    
    datasets = {
        "Линейно разделимые (Гауссианы)": generate_gaussian_clouds(noise_prob=0.05),
        "Нелинейные (XOR)": generate_xor(noise_prob=0.05),
        "Нелинейные (Окружности)": generate_circles(noise_prob=0.05)
    }
    
    results = {}
    for title, (X, y) in datasets.items():
        results[title] = evaluate_dataset(X, y, title)
        
    plot_combined_results(results)

# =========================================================
# БЛОК ВИЗУАЛИЗАЦИИ
# =========================================================

def plot_combined_results(results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Обучение перцептрона на линейных и нелинейных данных", fontsize=16, y=1.02)
    
    for col, (title, (model, X_test, y_test)) in enumerate(results.items()):
        
        ax_loss = axes[0, col]
        ax_loss.plot(model.history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        ax_loss.plot(model.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        ax_loss.set_title(f"Loss: {title}")
        ax_loss.set_xlabel('Эпохи')
        ax_loss.set_ylabel('Loss (BCE)')
        ax_loss.legend()
        ax_loss.grid(True, linestyle='--', alpha=0.7)

        ax_bound = axes[1, col]
        
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        
        # Предсказания
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # Отрисовка фона и точек
        ax_bound.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax_bound.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap='coolwarm')
        
        # Отрисовка разделяющей прямой
        w, b = model.w, model.b
        if w[1] != 0:
            x_line = np.linspace(x_min, x_max, 100)
            y_line = -(w[0] * x_line + b) / w[1]
    
            mask = (y_line >= y_min) & (y_line <= y_max)
            ax_bound.plot(x_line[mask], y_line[mask], color='black', linewidth=2)
        
        ax_bound.set_xlim(x_min, x_max)
        ax_bound.set_ylim(y_min, y_max)
            
        ax_bound.set_title(f"Граница: {title}")
        ax_bound.set_xlabel('X1')
        ax_bound.set_ylabel('X2')
        
    plt.tight_layout()
    os.makedirs("report/experiment_plots", exist_ok=True)
    plt.savefig("report/experiment_plots/linear_xor_circle_exp.jpg", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_generators_experiment()