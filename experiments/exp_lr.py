import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import generate_data, StandardScaler, train_test_split_stratified
from src.perceptron import Perceptron
from src.metrics import accuracy_score

def run_lr_experiment():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    learning_rates = [0.001, 0.01, 0.5, 1.0]
    histories = {}
    
    print(f"{'LR':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 35)
    
    for lr in learning_rates:
        model = Perceptron(init_type='small_random')
        model.fit(
            X_train_scaled, 
            y_train, 
            X_test_scaled, 
            y_test, 
            epochs=100, 
            lr=lr, 
            batch_size=32
        )
        
        histories[f'Learning Rate = {lr}'] = model.history
        acc_train = accuracy_score(y_train, model.predict(X_train_scaled))
        acc_test = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"{lr:<10} | {acc_train:<10.4f} | {acc_test:<10.4f}")

    plot_lr_results(histories)

# =========================================================
# БЛОК ВИЗУАЛИЗАЦИИ
# =========================================================

def plot_lr_results(histories):
    n = len(histories)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 5))
    fig.suptitle("Влияние скорости обучения (Learning Rate)", fontsize=16, y=1.05)
    
    for ax, (label, history) in zip(axes[:-1], histories.items()):
        ax.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        ax.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        ax.set_title(label)
        ax.set_xlabel('Эпохи')
        ax.set_ylabel('Loss (BCE)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
    ax_combined = axes[-1]
    for label, history in histories.items():
        ax_combined.plot(history['val_loss'], label=label, linewidth=2)
    ax_combined.set_title("Сводный график (Val Loss)")
    ax_combined.set_xlabel('Эпохи')
    ax_combined.legend()
    ax_combined.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs("report/experiment_plots", exist_ok=True)
    plt.savefig("report/experiment_plots/learning_rate_exp.jpg", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_lr_experiment()