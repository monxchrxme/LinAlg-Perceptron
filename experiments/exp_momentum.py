import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import generate_data, StandardScaler, train_test_split_stratified
from src.perceptron import Perceptron
from src.metrics import accuracy_score
from sklearn.datasets import make_classification

def run_momentum_experiment():
    # Генерируем данные и усложняем задачу (шум 5%)
    X, y = make_classification(
        n_samples=500, 
        n_features=2, n_redundant=0, 
        n_informative=2, 
        random_state=42, 
        n_clusters_per_class=1, 
        # flip_y=0.05
    )
    
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    betas = [0.0, 0.5, 0.9, 0.99]
    histories = {}
    
    print(f"{'Алгоритм':<20} | {'Test Acc':<10}")
    print("-" * 35)
    
    for beta in betas:
        model = Perceptron(init_type='small_random')
        # Ставим lr = 0.01, чтобы наглядно увидеть разницу в скорости
        model.fit(
            X_train_scaled, 
            y_train, 
            X_test_scaled, 
            y_test, 
            epochs=100, 
            lr=0.01, 
            batch_size=32, 
            loss_type='bce', 
            beta=beta
        )

        if beta == 0.0:
            label = "SGD (β=0.0)"
        else:
            label = f"Momentum (β={beta})"
        
        histories[label] = model.history
        acc_test = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"{label:<20} | {acc_test:<10.4f}")

    plot_momentum_results(histories)

def plot_momentum_results(histories):
    n = len(histories)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 5))
    fig.suptitle("Исследование сходимости: Обычный SGD против Momentum (lr=0.01)", fontsize=16, y=1.05)
    
    # Индивидуальные графики
    for ax, (label, history) in zip(axes[:-1], histories.items()):
        ax.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        ax.plot(history['val_loss'], label='Val Loss', color='orange', linewidth=2)
        ax.set_title(label)
        ax.set_xlabel('Эпохи')
        ax.set_ylabel('Loss (BCE)')
        ax.set_ylim([0.2, 0.8])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
    # Сводный график
    ax_combined = axes[-1]
    for label, history in histories.items():
        if "SGD" in label:
            ax_combined.plot(history['val_loss'], label=label, linewidth=3, linestyle='--', color='black')
        else:
            ax_combined.plot(history['val_loss'], label=label, linewidth=2)
            
    ax_combined.set_title("Сводный график (Val Loss)")
    ax_combined.set_xlabel('Эпохи')
    ax_combined.set_ylim([0.2, 0.8])
    ax_combined.legend()
    ax_combined.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    import os
    os.makedirs("report/experiment_plots", exist_ok=True)
    plt.savefig("report/experiment_plots/momentum_exp.jpg", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_momentum_experiment()