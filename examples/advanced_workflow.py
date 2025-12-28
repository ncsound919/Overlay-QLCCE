"""
Advanced example demonstrating full QLCCE workflow
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qlcce.core import (
    QuantumFieldSampler,
    MultiLogTransformer,
    ChaoticConstraintSystem,
    BenfordQuantumAnalyzer
)
import numpy as np
from scipy.integrate import odeint


def main():
    print("QLCCE Advanced Example: Complete Workflow")
    print("=" * 60)
    
    # === STEP 1: Quantum Field Generation ===
    print("\n[1] Quantum Field Generation")
    print("-" * 60)
    
    field_sampler = QuantumFieldSampler(
        lattice_size=64,
        mass=0.1,
        coupling=1.0
    )
    
    print("Generating quantum scalar field (this may take a moment)...")
    field = field_sampler.generate_scalar_field(steps=1000, temperature=1.0)
    
    # Calculate field properties
    properties = field_sampler.calculate_vacuum_fluctuations()
    print(f"Field shape: {field.shape}")
    print(f"Mean field value: {properties['mean']:.6f}")
    print(f"Variance: {properties['variance']:.6f}")
    print(f"Skewness: {properties['skewness']:.6f}")
    print(f"Kurtosis: {properties['kurtosis']:.6f}")
    print(f"Shannon entropy: {properties['entropy']:.4f} bits")
    
    # FFT analysis
    ft_power, kx, ky = field_sampler.fourier_transform_field()
    print(f"Fourier transform power spectrum computed")
    
    # === STEP 2: Multi-Base Logarithmic Analysis ===
    print("\n[2] Multi-Base Logarithmic Analysis")
    print("-" * 60)
    
    transformer = MultiLogTransformer()
    flat_field = np.abs(field.flatten())
    flat_field = flat_field[flat_field > 0]
    
    # Transform with multiple bases
    bases = ['e', '10', '2', 'golden', 'silver']
    log_results, divergence_matrix = transformer.multi_base_transform(
        flat_field,
        bases=bases
    )
    
    print(f"Transformed field with {len(bases)} different bases:")
    for base in bases:
        log_data = log_results[base]
        print(f"  Base {base:8s}: mean={np.mean(log_data):8.4f}, std={np.std(log_data):8.4f}")
    
    print(f"\nDivergence matrix (shows similarity between transformations):")
    print(divergence_matrix)
    
    # Detect log-periodicity
    print(f"\nDetecting log-periodic patterns...")
    periods, significance = transformer.detect_log_periodicity(flat_field, base='e')
    print(f"Found {len(periods)} significant periods")
    if len(periods) > 0:
        print(f"Top 3 periods: {periods[:3]}")
        print(f"Significance: {significance:.4f}")
    
    # === STEP 3: Comprehensive Benford Analysis ===
    print("\n[3] Benford's Law Compliance Analysis")
    print("-" * 60)
    
    analyzer = BenfordQuantumAnalyzer()
    
    # First digit analysis
    result_1st = analyzer.analyze_dataset(flat_field, digit_position=1)
    if result_1st:
        print(f"First Digit Analysis:")
        print(f"  Compliance score: {result_1st['compliance_score']:.4f}")
        print(f"  Chi-square p-value: {result_1st['chi2_p_value']:.6f}")
        print(f"  KS statistic: {result_1st['ks_statistic']:.6f}")
        print(f"  KL divergence: {result_1st['kl_divergence']:.6f}")
        print(f"  JS divergence: {result_1st['js_divergence']:.6f}")
    
    # Multiscale analysis
    print(f"\nMultiscale Benford Analysis:")
    multiscale_results = analyzer.analyze_multiscale(flat_field, scales=[1, 10, 50])
    for scale_key, scale_data in multiscale_results.items():
        print(f"  {scale_key}: avg_compliance={scale_data['avg_compliance']:.4f}, n_bins={scale_data['n_bins']}")
    
    # Quantum-Benford correlation
    print(f"\nQuantum Field - Benford Correlation:")
    qb_corr = analyzer.quantum_benford_correlation(field)
    if qb_corr:
        print(f"  Mean regional compliance: {qb_corr['mean_compliance']:.4f}")
        print(f"  Std regional compliance: {qb_corr['std_compliance']:.4f}")
        print(f"  Regional compliance values: {qb_corr['region_compliance']}")
    
    # === STEP 4: Chaotic Systems with Constraints ===
    print("\n[4] Chaotic Systems with Adaptive Constraints")
    print("-" * 60)
    
    chaos = ChaoticConstraintSystem(system_type='lorenz')
    
    # Add multiple constraints
    print("Adding constraints:")
    boundary_const = chaos.add_constraint('boundary', bounds=[(-20, 20), (-30, 30), (0, 50)])
    print("  ✓ Boundary constraint (reflecting walls)")
    
    quantum_const = chaos.add_constraint('quantum', uncertainty=0.05)
    print("  ✓ Quantum uncertainty constraint")
    
    symmetry_const = chaos.add_constraint('symmetry', axis='z')
    print("  ✓ Symmetry constraint (z-axis reflection)")
    
    # Generate Lorenz trajectory
    print("\nGenerating constrained Lorenz trajectory...")
    system_eq = chaos.lorenz_system(constraints=chaos.constraints)
    t = np.linspace(0, 50, 5000)
    initial_state = [1.0, 1.0, 1.0]
    trajectory = odeint(system_eq, initial_state, t)
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"State space bounds:")
    print(f"  X: [{trajectory[:, 0].min():8.2f}, {trajectory[:, 0].max():8.2f}]")
    print(f"  Y: [{trajectory[:, 1].min():8.2f}, {trajectory[:, 1].max():8.2f}]")
    print(f"  Z: [{trajectory[:, 2].min():8.2f}, {trajectory[:, 2].max():8.2f}]")
    
    # Calculate Lyapunov exponents
    print("\nCalculating Lyapunov exponents...")
    lyap_exp = chaos.calculate_lyapunov_exponents(trajectory[:, 0], dt=0.01)
    print(f"Lyapunov exponents: {lyap_exp}")
    if np.any(lyap_exp > 0):
        print("  ✓ System exhibits chaotic behavior (positive exponent)")
    
    # === STEP 5: Cross-System Analysis ===
    print("\n[5] Cross-System Integration")
    print("-" * 60)
    
    # Analyze Benford compliance of chaotic trajectory
    print("Benford analysis of chaotic trajectory:")
    trajectory_flat = np.abs(trajectory.flatten())
    trajectory_flat = trajectory_flat[trajectory_flat > 0]
    
    chaos_benford = analyzer.analyze_dataset(trajectory_flat)
    if chaos_benford:
        print(f"  Compliance score: {chaos_benford['compliance_score']:.4f}")
        print(f"  Chi-square p-value: {chaos_benford['chi2_p_value']:.6f}")
    
    # Compare with quantum field Benford compliance
    if result_1st and chaos_benford:
        diff = abs(result_1st['compliance_score'] - chaos_benford['compliance_score'])
        print(f"\nBenford compliance comparison:")
        print(f"  Quantum field: {result_1st['compliance_score']:.4f}")
        print(f"  Chaotic system: {chaos_benford['compliance_score']:.4f}")
        print(f"  Difference: {diff:.4f}")
    
    # === Summary ===
    print("\n" + "=" * 60)
    print("Advanced Analysis Complete!")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"  • Quantum field generated with {field.size} lattice points")
    print(f"  • {len(bases)} logarithmic bases analyzed")
    print(f"  • Benford compliance scores calculated across multiple scales")
    print(f"  • Chaotic system with {len(chaos.constraints)} constraints evolved")
    print(f"  • Cross-system patterns identified")
    print("\nThis demonstrates the full power of QLCCE for integrated")
    print("quantum-chaotic-logarithmic research workflows!")


if __name__ == "__main__":
    main()
