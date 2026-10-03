import numpy as np
from tensor_transform import flatten_tensor, reconstruct_tensor

def compute_errors(original, reconstructed):
    """
    Computes E_max and MAE.
    E_max = max | I - I_hat |
    MAE = 1 / (BCHW) * sum | I - I_hat |
    """
    diff = np.abs(original - reconstructed)
    e_max = np.max(diff)
    mae = np.mean(diff)
    return e_max, mae

def run_experiment(name, B, C, H, W):
    print(f"Running experiment for {name} with shape B={B}, C={C}, H={H}, W={W}...")
    # Generate random synthetic data for this tensor shape
    tensor = np.random.rand(B, C, H, W).astype(np.float32)
    
    # 1. Flatten
    flattened = flatten_tensor(tensor)
    
    # 2. Reconstruct
    reconstructed = reconstruct_tensor(flattened, (B, C, H, W))
    
    # 3. Compare
    e_max, mae = compute_errors(tensor, reconstructed)
    
    return B, C, H, W, e_max, mae

def main():
    experiments = [
        ("MNIST / sample", 1, 1, 28, 28),
        ("CIFAR / sample", 1, 3, 32, 32),
        ("Synthetic fmap1", 2, 16, 32, 32),
        ("Synthetic fmap2", 2, 64, 16, 16),
        ("Synthetic fmap3", 2, 128, 8, 8),
        ("Synthetic fmap4", 2, 500, 4, 4),
    ]
    
    print("-" * 80)
    print(f"{'Dataset/Input':<18} | {'B':<3} | {'C':<4} | {'H':<3} | {'W':<3} | {'Emax':<8} | {'MAE':<8}")
    print("-" * 80)
    
    results = []
    for exp in experiments:
        B, C, H, W, e_max, mae = run_experiment(*exp)
        results.append((exp[0], B, C, H, W, e_max, mae))
        
    print("\n\nFinal Table:")
    print("-" * 80)
    print(f"{'Dataset/Input':<18} | {'B':<3} | {'C':<4} | {'H':<3} | {'W':<3} | {'Emax':<8} | {'MAE':<8}")
    print("-" * 80)
    for res in results:
        name, B, C, H, W, e_max, mae = res
        print(f"{name:<18} | {B:<3} | {C:<4} | {H:<3} | {W:<3} | {e_max:<8.2e} | {mae:<8.2e}")
    print("-" * 80)

if __name__ == "__main__":
    main()
