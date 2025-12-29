"""
Example script showing basic usage of QLCCE
"""

import sys
import os
# Add parent directory to path for development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qlcce.core import (
    QuantumFieldSampler,
    MultiLogTransformer,
    ChaoticConstraintSystem,
    BenfordQuantumAnalyzer
)
import numpy as np


def main():
    print("QLCCE Example: Basic Usage")
    print("=" * 50)
    
    # 1. Create a quantum field
    print("\n1. Generating quantum field...")
    field_sampler = QuantumFieldSampler(lattice_size=32, mass=0.1, coupling=1.0)
    field = field_sampler.generate_scalar_field(steps=500)
    properties = field_sampler.calculate_vacuum_fluctuations()
    print(f"   Field shape: {field.shape}")
    print(f"   Mean: {properties['mean']:.6f}")
    print(f"   Variance: {properties['variance']:.6f}")
    
    # 2. Apply logarithmic transformations
    print("\n2. Applying logarithmic transformations...")
    transformer = MultiLogTransformer()
    flat_field = np.abs(field.flatten())
    flat_field = flat_field[flat_field > 0]
    
    log_e = transformer.transform(flat_field, base='e', mode='direct')
    log_10 = transformer.transform(flat_field, base='10', mode='direct')
    print(f"   Log base e: mean={np.mean(log_e):.4f}, std={np.std(log_e):.4f}")
    print(f"   Log base 10: mean={np.mean(log_10):.4f}, std={np.std(log_10):.4f}")
    
    # 3. Analyze Benford's law compliance
    print("\n3. Analyzing Benford's law compliance...")
    analyzer = BenfordQuantumAnalyzer()
    result = analyzer.analyze_dataset(flat_field)
    if result:
        print(f"   Compliance score: {result['compliance_score']:.4f}")
        print(f"   Chi-square p-value: {result['chi2_p_value']:.6f}")
        print(f"   First 3 digit frequencies: {result['digit_frequencies'][:3]}")
    
    # 4. Create a chaotic system
    print("\n4. Running chaotic system...")
    chaos = ChaoticConstraintSystem(system_type='lorenz')
    
    # Add a boundary constraint
    chaos.add_constraint('boundary', bounds=[(-20, 20), (-30, 30), (0, 50)])
    
    # Generate Lorenz system
    from scipy.integrate import odeint
    system_eq = chaos.lorenz_system(constraints=chaos.constraints)
    t = np.linspace(0, 50, 5000)
    trajectory = odeint(system_eq, [1.0, 1.0, 1.0], t)
    print(f"   Generated trajectory: {trajectory.shape}")
    print(f"   X range: [{trajectory[:, 0].min():.2f}, {trajectory[:, 0].max():.2f}]")
    print(f"   Y range: [{trajectory[:, 1].min():.2f}, {trajectory[:, 1].max():.2f}]")
    print(f"   Z range: [{trajectory[:, 2].min():.2f}, {trajectory[:, 2].max():.2f}]")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()
