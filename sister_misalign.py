

import numpy as np
import matplotlib.pyplot as plt

def crosslink_chains_gaussian_misalign(N_crosslinks, sigma_offset, path, iteration, rng=None, plot_hist=True):
    if rng is None:
        rng = np.random.default_rng()
    
    N_monomers = 32000
    # Sample anchor positions on first chain (without replacement)
    anchor_positions = rng.choice(np.arange(N_monomers), size=N_crosslinks, replace=False)
    # Generate Gaussian misalignments and round to lattice positions
    raw_misalignment = rng.normal(0, sigma_offset, size=N_crosslinks)
    lattice_misalignment = np.round(raw_misalignment).astype(int)
    # For each crosslink, randomly choose which leg to move (50/50 chance)
    move_first_chain = rng.random(N_crosslinks) < 0.5
    # Initialize bond array: [chain1_pos, chain2_pos]
    bond_array = np.zeros((N_crosslinks, 2), dtype=int)
    bond_array[:, 0] = anchor_positions  # First chain positions
    bond_array[:, 1] = anchor_positions + N_monomers  # Second chain positions (initially aligned)
    
    # Apply misalignments
    for i in range(N_crosslinks):
        if move_first_chain[i]:
            # Move first chain leg
            new_pos = anchor_positions[i] + lattice_misalignment[i]
            bond_array[i, 0] = np.clip(new_pos, 0, N_monomers - 1)
        else:
            # Move second chain leg
            new_pos = anchor_positions[i] + N_monomers + lattice_misalignment[i]
            bond_array[i, 1] = np.clip(new_pos, N_monomers, 2 * N_monomers - 1)
    
    sister1_sorted = np.sort(bond_array[:, 0])
    sister2_sorted = np.sort(bond_array[:, 1])
    # Calculate actual misalignments for analysis
    actual_misalignment = np.where(move_first_chain, 
                                 bond_array[:, 0] - anchor_positions,
                                 bond_array[:, 1] - (anchor_positions + N_monomers))
    
    # Save files - FIXED: save full arrays, not just last element
    # np.savetxt(f'{path}/anchor_{N_crosslinks}_{iteration}_sister1.txt', bond_array[:, 0])
    # np.savetxt(f'{path}/anchor_{N_crosslinks}_{iteration}_sister2.txt', bond_array[:, 1])
    
    # Plot histograms if requested
    if plot_hist:
        plot_misalignment_analysis(anchor_positions, bond_array, actual_misalignment, 
                                 sigma_offset, N_monomers, N_crosslinks)
    
    # Update topology
    # extrude.update_topology(system, bond_array, bond_name='Sister')
    
    return {
        'anchor_positions': anchor_positions,
        'bond_array': bond_array,
        'actual_misalignment': actual_misalignment,
        'moved_first_chain': move_first_chain
    }

def plot_misalignment_analysis(anchor_positions, bond_array, actual_misalignment, 
                             sigma_offset, N_monomers, N_crosslinks):
    """Plot histograms of misalignments and chain positions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Histogram of misalignments
    axes[0, 0].hist(actual_misalignment, alpha=0.7, edgecolor='black', color='green')
    axes[0, 0].set_xlabel('Misalignment (monomers)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Misalignment Distribution\n(σ={sigma_offset}, actual std={np.std(actual_misalignment):.2f})')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='No misalignment')
    axes[0, 0].legend()
    
    # 2. Histogram of Chain 1 positions
    axes[0, 1].hist(bond_array[:, 0], bins=50, alpha=0.7, edgecolor='black', color='blue')
    axes[0, 1].set_xlabel('Chain 1 Position')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Chain 1 Crosslink Positions\n(range: {bond_array[:, 0].min()}-{bond_array[:, 0].max()})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram of Chain 2 positions
    axes[1, 0].hist(bond_array[:, 1], bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel('Chain 2 Position')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Chain 2 Crosslink Positions\n(range: {bond_array[:, 1].min()}-{bond_array[:, 1].max()})')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(N_monomers, color='red', linestyle='--', alpha=0.7, label='Chain 2 start')
    axes[1, 0].legend()
    
    # 4. Scatter plot: Chain 1 vs Chain 2 positions
    axes[1, 1].scatter(bond_array[:, 0], bond_array[:, 1], alpha=0.6, s=20)
    axes[1, 1].plot([0, N_monomers-1], [N_monomers, 2*N_monomers-1], 'r--', alpha=0.7, label='Perfect alignment')
    axes[1, 1].set_xlabel('Chain 1 Position')
    axes[1, 1].set_ylabel('Chain 2 Position')
    axes[1, 1].set_title('Chain 1 vs Chain 2 Crosslink Positions')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nCrosslink Statistics:")
    print(f"Total crosslinks: {N_crosslinks}")
    print(f"Target σ: {sigma_offset}")
    print(f"Actual misalignment std: {np.std(actual_misalignment):.3f}")
    print(f"Zero misalignment fraction: {np.mean(actual_misalignment == 0):.2%}")
    print(f"Chain 1 range: {bond_array[:, 0].min()} - {bond_array[:, 0].max()}")
    print(f"Chain 2 range: {bond_array[:, 1].min()} - {bond_array[:, 1].max()}")

# Test function
def test_crosslink_function():
    """Test the crosslink function with mock system."""

    
    rng = np.random.default_rng(42)
    
    # Test with σ = 1.5
    result = crosslink_chains_gaussian_misalign(
        N_crosslinks=1000, sigma_offset=50, 
        path='test', iteration=1, rng=rng, plot_hist=True
    )
    
    return result

if __name__ == "__main__":
    test_crosslink_function()
