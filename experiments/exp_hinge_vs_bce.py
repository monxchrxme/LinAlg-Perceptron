import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import generate_data, StandardScaler, train_test_split_stratified
from src.perceptron import Perceptron
from src.metrics import accuracy_score

def run_hinge_experiment():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    losses_to_test = ['bce', 'hinge']
    histories = {}
    
    print(f"{'Loss Type':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 35)
    
    for l_type in losses_to_test:
        model = Perceptron(init_type='small_random')
        model.fit(
            X_train_scaled, 
            y_train, 
            X_test_scaled, 
            y_test, 
            epochs=100, 
            lr=0.1, 
            batch_size=32, 
            loss_type=l_type
        )
        
        histories[l_type] = model.history
        acc_train = accuracy_score(y_train, model.predict(X_train_scaled))
        acc_test = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"{l_type:<10} | {acc_train:<10.4f} | {acc_test:<10.4f}")

    plot_hinge_results(histories)

# =========================================================
# БЛОК ВИЗУАЛИЗАЦИИ 
# =========================================================

def plot_hinge_results(histories):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Сравнение BCE и Hinge Loss", fontsize=16, y=1.05)
    
    # Индивидуальные графики 
    for ax, (label, history) in zip(axes[:-1], histories.items()):
        ax.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        ax.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        ax.set_title(f"Функция потерь: {label.upper()}")
        ax.set_xlabel('Эпохи')
        ax.set_ylabel('Loss Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
    # Сводный график 
    ax_combined = axes[-1]
    for label, history in histories.items():
        ax_combined.plot(history['val_loss'], label=f"{label.upper()} (Val)", linewidth=2)
    ax_combined.set_title("Сводный график (Скорость сходимости)")
    ax_combined.set_xlabel('Эпохи')
    ax_combined.set_ylabel('Loss Value')
    ax_combined.legend()
    ax_combined.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs("report/experiment_plots", exist_ok=True)
    plt.savefig("report/experiment_plots/exp_hinge_vs_bce.jpg", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_hinge_experiment()