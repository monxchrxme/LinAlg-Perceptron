import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import generate_data, StandardScaler, train_test_split_stratified
from src.perceptron import Perceptron
from src.metrics import accuracy_score

def run_l2_experiment():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    l2_coefs = [0.0, 0.01, 0.1, 1.0]
    histories = {}
    
    print(f"{'L2 Coef (λ)':<12} | {'Test Acc':<10} | {'Норма весов ||w||':<20}")
    print("-" * 50)
    
    for l2 in l2_coefs:
        model = Perceptron(init_type='small_random')
        model.fit(
            X_train_scaled, 
            y_train, 
            X_test_scaled, 
            y_test, 
            epochs=100, 
            lr=0.1, 
            batch_size=32, 
            loss_type='bce', 
            l2_coef=l2)
        
        histories[f'λ = {l2}'] = model.history
        acc_test = accuracy_score(y_test, model.predict(X_test_scaled))
        
        # Считаем длину вектора весов (Евклидова норма)
        weight_norm = np.linalg.norm(model.w)
        print(f"{l2:<12} | {acc_test:<10.4f} | {weight_norm:<20.4f}")

    plot_l2_results(histories)

def plot_l2_results(histories):
    n = len(histories)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 5))
    fig.suptitle("Влияние L2-регуляризации", fontsize=16, y=1.05)
    
    for ax, (label, history) in zip(axes[:-1], histories.items()):
        ax.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        ax.plot(history['val_loss'], label='Val Loss', color='orange', linewidth=2)
        ax.set_title(label)
        ax.set_xlabel('Эпохи')
        ax.set_ylabel('Loss (BCE + L2)')
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
    plt.savefig("report/experiment_plots/l2_exp.jpg", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_l2_experiment()