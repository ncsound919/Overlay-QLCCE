#!/usr/bin/env python3
"""
QLCCE Command Line Interface
"""

import argparse
import json
import sys


def main():
    """Command line interface for QLCCE"""
    from datetime import datetime
    import numpy as np
    from scipy import integrate, stats
    import matplotlib.pyplot as plt
    from qlcce.core import (
        QuantumFieldSampler,
        MultiLogTransformer,
        ChaoticConstraintSystem,
        BenfordQuantumAnalyzer
    )
    
    parser = argparse.ArgumentParser(
        description='Quantum-Logarithmic Chaotic Constraint Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick            # Quick analysis with default settings
  %(prog)s --full --steps 2000  # Full analysis with 2000 steps
  %(prog)s --config my_config.json  # Load configuration from file
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis (default settings)')
    parser.add_argument('--full', action='store_true',
                       help='Run full analysis with visualization')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of simulation steps')
    parser.add_argument('--config', type=str,
                       help='JSON configuration file')
    parser.add_argument('--export', type=str,
                       help='Export results to specified file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'lattice_size': 64 if not args.quick else 32,
            'field_mass': 0.1,
            'field_coupling': 1.0,
            'chaos_system': 'lorenz',
            'log_bases': ['e', '10', '2', 'golden'],
            'constraints': ['boundary', 'quantum']
        }
    
    print("=" * 60)
    print("QLCCE: Quantum-Logarithmic Chaotic Constraint Engine")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now()}")
    print(f"Configuration: {config}")
    
    # Initialize components
    quantum_field = QuantumFieldSampler(
        lattice_size=config['lattice_size'],
        mass=config['field_mass'],
        coupling=config['field_coupling']
    )
    
    log_transformer = MultiLogTransformer()
    chaos_system = ChaoticConstraintSystem(system_type=config['chaos_system'])
    benford_analyzer = BenfordQuantumAnalyzer()
    
    results = {}
    
    # Step 1: Generate quantum field
    print("\n1. Generating quantum scalar field...")
    n_steps = 500 if args.quick else args.steps
    field = quantum_field.generate_scalar_field(steps=n_steps)
    field_properties = quantum_field.calculate_vacuum_fluctuations()
    
    results['field'] = field
    results['field_properties'] = field_properties
    print(f"   Field generated: {field.shape}")
    print(f"   Mean: {field_properties['mean']:.6f}")
    print(f"   Variance: {field_properties['variance']:.6f}")
    
    # Step 2: Apply logarithmic transformations
    print("\n2. Applying multi-base logarithmic transforms...")
    flat_field = np.abs(field.flatten())
    flat_field = flat_field[flat_field > 0]  # Filter positive values
    log_results, divergence_matrix = log_transformer.multi_base_transform(
        flat_field, 
        bases=config['log_bases']
    )
    
    results['log_transforms'] = log_results
    results['log_divergence'] = divergence_matrix
    print(f"   Transformed with bases: {config['log_bases']}")
    
    # Step 3: Benford's law analysis
    print("\n3. Analyzing Benford's law compliance...")
    benford_result = benford_analyzer.analyze_dataset(flat_field)
    
    if benford_result:
        results['benford'] = benford_result
        print(f"   Benford compliance score: {benford_result['compliance_score']:.4f}")
        print(f"   Chi-square p-value: {benford_result['chi2_p_value']:.6f}")
        print(f"   KL divergence: {benford_result['kl_divergence']:.6f}")
    
    # Step 4: Run chaotic system
    print("\n4. Running chaotic system with constraints...")
    
    # Add constraints
    constraints = []
    for constraint_type in config.get('constraints', []):
        if constraint_type == 'boundary':
            constraint = chaos_system.add_constraint('boundary', 
                                                    bounds=[(-20, 20), (-30, 30), (0, 50)])
            constraints.append(constraint)
        elif constraint_type == 'quantum':
            constraint = chaos_system.add_constraint('quantum', uncertainty=0.05)
            constraints.append(constraint)
    
    # Generate chaotic trajectory
    from scipy.integrate import odeint
    system_eq = chaos_system.lorenz_system(constraints=chaos_system.constraints)
    t = np.linspace(0, 50, 5000)
    initial_state = [1.0, 1.0, 1.0]
    trajectory = odeint(system_eq, initial_state, t)
    
    results['chaotic_trajectory'] = trajectory
    print(f"   Generated trajectory with {len(constraints)} constraints")
    print(f"   Trajectory shape: {trajectory.shape}")
    
    # Export if requested
    if args.export:
        export_file = args.export
    elif args.full:
        export_file = f"qlcce_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        export_file = None
    
    if export_file:
        print(f"\n5. Exporting results to {export_file}...")
        # Convert numpy arrays to lists for JSON serialization
        export_data = {
            'timestamp': str(datetime.now()),
            'config': config,
            'field_properties': field_properties,
            'benford_compliance': benford_result['compliance_score'] if benford_result else None,
            'chi2_p_value': benford_result['chi2_p_value'] if benford_result else None
        }
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"   Results exported successfully")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
