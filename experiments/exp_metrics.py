import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import generate_data, StandardScaler, train_test_split_stratified
from src.perceptron import Perceptron
from src.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

def run_metrics_experiment():
    print("=== Обучение модели и анализ метрик ===")
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=500, 
        n_features=2, 
        n_redundant=0, 
        n_informative=2, 
        random_state=42, 
        n_clusters_per_class=1, 
        # flip_y=0.05
    )
                               
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Perceptron(init_type='small_random')
    model.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, lr=0.1, batch_size=32)
    
    # Жесткие предсказания (0 или 1) для метрик
    y_pred = model.predict(X_test_scaled)
    # Вероятности (от 0 до 1) для ROC-кривой (метод forward возвращает вероятности)
    y_probs = model.forward(X_test_scaled)
   
    print("\n=== Метрики на тестовой выборке ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_probs):.4f}")
    
    plot_metrics_analysis(model, X_test_scaled, y_test, y_pred, y_probs)

def plot_metrics_analysis(model, X_test, y_test, y_pred, y_probs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Анализ ошибок и ROC-кривая", fontsize=16)

    # Левый граифк: ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (FPR)')
    axes[0].set_ylabel('True Positive Rate (TPR)')
    axes[0].set_title('ROC-кривая')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Правый график: Анализ ошибок на границе
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    w, b = model.w, model.b
    if w[1] != 0:
        x_line = np.linspace(x_min, x_max, 100)
        y_line = -(w[0] * x_line + b) / w[1]
        mask = (y_line >= y_min) & (y_line <= y_max)
        axes[1].plot(x_line[mask], y_line[mask], color='black', linewidth=2, label='Разделяющая прямая')
        
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    correct = (y_test == y_pred)
    incorrect = (y_test != y_pred)
    
    # Рисуем правильные точки (кружочки)
    axes[1].scatter(X_test[correct, 0], X_test[correct, 1], c=y_test[correct], 
                    marker='o', edgecolors='k', cmap='coolwarm', alpha=0.7, label='Верно')
    
    # Рисуем ошибки (большие крестики с обводкой)
    axes[1].scatter(X_test[incorrect, 0], X_test[incorrect, 1], c=y_test[incorrect], 
                    marker='X', s=120, edgecolors='yellow', linewidth=1.5, cmap='coolwarm', label='ОШИБКА')

    axes[1].set_title("Визуализация ошибочных классификаций")
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("report/experiment_plots", exist_ok=True)
    plt.savefig("report/experiment_plots/metrics_exp.jpg", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_metrics_experiment()