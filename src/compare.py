import matplotlib.pyplot as plt
import os

def read_results(filepath):
    epochs, train_losses, train_accs, test_accs = [], [], [], []
    with open(filepath) as f:
        lines = f.readlines()[1:]
        for line in lines:
            e, tl, ta, tea = line.strip().split(",")
            epochs.append(int(e))
            train_losses.append(float(tl))
            train_accs.append(float(ta))
            test_accs.append(float(tea))
    return epochs, train_losses, train_accs, test_accs

def read_vae_loss(filepath):
    epochs, losses = [], []
    with open(filepath) as f:
        lines = f.readlines()[1:]
        for line in lines:
            e, l = line.strip().split(",")
            epochs.append(int(e))
            losses.append(float(l))
    return epochs, losses

os.makedirs("results", exist_ok=True)

# VAE loss curve
if os.path.exists("results/exp1_vae_loss.txt"):
    epochs, vae_losses = read_vae_loss("results/exp1_vae_loss.txt")
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, vae_losses)
    plt.xlabel("Epoch")
    plt.ylabel("VAE Loss")
    plt.title("Phase 1: VAE Unsupervised Training Loss")
    plt.grid(True)
    plt.savefig("results/vae_loss_curve.png")
    print("Saved results/vae_loss_curve.png")

# Classification comparison
exp_names = {
    "results/exp1_results.txt": "Exp1: Frozen Encoder",
    "results/exp2_results.txt": "Exp2: Fine-tune Encoder",
    "results/exp3_results.txt": "Exp3: Standalone Classifier",
}

files = [f for f in exp_names if os.path.exists(f)]

if files:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for filepath in files:
        epochs, train_losses, train_accs, test_accs = read_results(filepath)
        label = exp_names[filepath]

        axes[0].plot(epochs, train_losses, label=label)
        axes[1].plot(epochs, train_accs, label=label)
        axes[2].plot(epochs, test_accs, label=label)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss Comparison")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train Accuracy (%)")
    axes[1].set_title("Training Accuracy Comparison")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Test Accuracy (%)")
    axes[2].set_title("Test Accuracy Comparison")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("results/comparison.png")
    print("Saved results/comparison.png")

    # Print final accuracy table
    print()
    print("=" * 60)
    print(f"{'Experiment':<30} {'Train Acc':<12} {'Test Acc':<12}")
    print("=" * 60)
    for filepath in files:
        _, _, train_accs, test_accs = read_results(filepath)
        print(f"{exp_names[filepath]:<30} {train_accs[-1]:<12.2f} {test_accs[-1]:<12.2f}")
    print("=" * 60)
else:
    print("No result files found. Run the experiments first.")
