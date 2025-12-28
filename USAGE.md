# QLCCE Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Command Line Interface](#command-line-interface)
5. [Python API](#python-api)
6. [Examples](#examples)

## Installation

### Requirements
- Python >= 3.7
- numpy >= 1.21.0
- scipy >= 1.7.0
- numba >= 0.54.0
- sympy >= 1.9
- matplotlib >= 3.4.0

### Install from source

```bash
git clone https://github.com/ncsound919/Overlay-QLCCE.git
cd Overlay-QLCCE
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line

```bash
# Quick analysis
python -m qlcce.cli --quick

# Full analysis with visualization
python -m qlcce.cli --full --steps 2000

# Custom configuration
python -m qlcce.cli --config configs/default.json --export results.json
```

### Python Script

```python
from qlcce.core import QuantumFieldSampler, BenfordQuantumAnalyzer
import numpy as np

# Generate quantum field
sampler = QuantumFieldSampler(lattice_size=64)
field = sampler.generate_scalar_field(steps=1000)

# Analyze Benford's law
analyzer = BenfordQuantumAnalyzer()
result = analyzer.analyze_dataset(np.abs(field.flatten()))
print(f"Compliance: {result['compliance_score']:.4f}")
```

## Core Components

### 1. QuantumFieldSampler

Simulates quantum field fluctuations using Metropolis-Hastings MCMC on a lattice.

**Key Methods:**
- `generate_scalar_field(steps, temperature)` - Generate field configuration
- `calculate_vacuum_fluctuations()` - Calculate statistical properties
- `fourier_transform_field()` - Compute momentum-space representation
- `measure_correlation_function(max_distance)` - Two-point correlations

**Example:**
```python
from qlcce.core import QuantumFieldSampler

sampler = QuantumFieldSampler(
    lattice_size=64,
    mass=0.1,
    coupling=1.0
)

field = sampler.generate_scalar_field(steps=1000, temperature=1.0)
properties = sampler.calculate_vacuum_fluctuations()

print(f"Mean: {properties['mean']}")
print(f"Variance: {properties['variance']}")
print(f"Entropy: {properties['entropy']} bits")
```

### 2. MultiLogTransformer

Multi-base logarithmic transformations with chaotic perturbations.

**Available Bases:**
- Standard: 'e', '10', '2'
- Special: 'golden' (φ), 'silver' (δₛ), 'bronze' (σ), 'pi', 'sqrt2', 'fibonacci'

**Transformation Modes:**
- `direct` - Standard logarithm
- `iterated` - Iterated logarithm log*(x)
- `super-log` - Inverse tetration
- `chaotic` - Logarithm with Lorenz perturbation

**Example:**
```python
from qlcce.core import MultiLogTransformer
import numpy as np

transformer = MultiLogTransformer()
data = np.random.exponential(scale=10, size=1000)

# Single transformation
log_e = transformer.transform(data, base='e', mode='direct')

# Multiple bases
results, divergence = transformer.multi_base_transform(
    data,
    bases=['e', '10', '2', 'golden']
)

# Detect log-periodicity
periods, significance = transformer.detect_log_periodicity(data, base='e')
```

### 3. ChaoticConstraintSystem

Chaotic dynamical systems with adaptive constraints.

**Supported Systems:**
- Lorenz attractor
- Rössler system
- Logistic map

**Constraint Types:**
- `boundary` - Reflecting/absorbing boundaries
- `symmetry` - Spatial symmetries
- `quantum` - Heisenberg uncertainty
- `conservation` - Conservation laws (placeholder)

**Example:**
```python
from qlcce.core import ChaoticConstraintSystem
from scipy.integrate import odeint
import numpy as np

chaos = ChaoticConstraintSystem(system_type='lorenz')

# Add constraints
chaos.add_constraint('boundary', bounds=[(-20, 20), (-30, 30), (0, 50)])
chaos.add_constraint('quantum', uncertainty=0.05)

# Evolve system
system_eq = chaos.lorenz_system(constraints=chaos.constraints)
t = np.linspace(0, 50, 5000)
trajectory = odeint(system_eq, [1.0, 1.0, 1.0], t)

# Calculate Lyapunov exponents
lyap = chaos.calculate_lyapunov_exponents(trajectory[:, 0])

# Detect bifurcations (for logistic map)
chaos_logistic = ChaoticConstraintSystem(system_type='logistic')
bifurcations = chaos_logistic.detect_bifurcations(
    parameter_range=[2.5, 4.0],
    n_points=100
)
```

### 4. BenfordQuantumAnalyzer

Advanced Benford's law analysis with quantum field correlations.

**Statistical Tests:**
- Chi-square goodness-of-fit
- Kolmogorov-Smirnov test
- Kullback-Leibler divergence
- Jensen-Shannon divergence

**Analysis Types:**
- First/nth digit distribution
- Multiscale analysis
- Quantum field correlation

**Example:**
```python
from qlcce.core import BenfordQuantumAnalyzer
import numpy as np

analyzer = BenfordQuantumAnalyzer()

# Basic analysis
data = np.random.exponential(scale=100, size=10000)
result = analyzer.analyze_dataset(data, digit_position=1)

print(f"Compliance score: {result['compliance_score']:.4f}")
print(f"Chi-square p-value: {result['chi2_p_value']:.6f}")
print(f"KL divergence: {result['kl_divergence']:.6f}")

# Multiscale analysis
multiscale = analyzer.analyze_multiscale(data, scales=[1, 10, 100])

# Quantum field correlation
from qlcce.core import QuantumFieldSampler
sampler = QuantumFieldSampler(lattice_size=64)
field = sampler.generate_scalar_field(steps=500)
qb_corr = analyzer.quantum_benford_correlation(field)
```

## Command Line Interface

### Usage

```
python -m qlcce.cli [options]
```

### Options

- `--quick` - Run quick analysis (32x32 lattice, 500 steps)
- `--full` - Run full analysis with visualization
- `--steps N` - Number of simulation steps (default: 1000)
- `--config FILE` - Load configuration from JSON file
- `--export FILE` - Export results to JSON file

### Configuration File Format

```json
{
  "lattice_size": 64,
  "field_mass": 0.1,
  "field_coupling": 1.0,
  "chaos_system": "lorenz",
  "log_bases": ["e", "10", "2", "golden"],
  "constraints": ["boundary", "quantum"]
}
```

### Example Commands

```bash
# Quick test
python -m qlcce.cli --quick

# Full analysis with 2000 steps
python -m qlcce.cli --full --steps 2000

# Custom config and export
python -m qlcce.cli --config my_config.json --export my_results.json

# Help
python -m qlcce.cli --help
```

## Python API

### Complete Workflow Example

```python
from qlcce.core import (
    QuantumFieldSampler,
    MultiLogTransformer,
    ChaoticConstraintSystem,
    BenfordQuantumAnalyzer
)
import numpy as np
from scipy.integrate import odeint

# 1. Generate quantum field
field_sampler = QuantumFieldSampler(lattice_size=64, mass=0.1, coupling=1.0)
field = field_sampler.generate_scalar_field(steps=1000)
field_props = field_sampler.calculate_vacuum_fluctuations()

# 2. Transform with multiple logarithmic bases
transformer = MultiLogTransformer()
flat_field = np.abs(field.flatten())
flat_field = flat_field[flat_field > 0]
log_results, divergence = transformer.multi_base_transform(flat_field)

# 3. Analyze Benford's law
analyzer = BenfordQuantumAnalyzer()
benford = analyzer.analyze_dataset(flat_field)
multiscale = analyzer.analyze_multiscale(flat_field)
qb_corr = analyzer.quantum_benford_correlation(field)

# 4. Run chaotic system
chaos = ChaoticConstraintSystem(system_type='lorenz')
chaos.add_constraint('boundary', bounds=[(-20, 20), (-30, 30), (0, 50)])
system_eq = chaos.lorenz_system(constraints=chaos.constraints)
t = np.linspace(0, 50, 5000)
trajectory = odeint(system_eq, [1.0, 1.0, 1.0], t)

# 5. Results
print(f"Field variance: {field_props['variance']:.4f}")
print(f"Benford compliance: {benford['compliance_score']:.4f}")
print(f"Chaotic trajectory range: {trajectory.min():.2f} to {trajectory.max():.2f}")
```

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py` - Simple usage of each component
- `advanced_workflow.py` - Complete research workflow with all features

Run examples:
```bash
python examples/basic_usage.py
python examples/advanced_workflow.py
```

## Tips and Best Practices

1. **Memory Management**: Large lattice sizes (>128) require significant RAM. Start with 32 or 64.

2. **Convergence**: Quantum field generation requires adequate steps for convergence. Use at least 1000 steps.

3. **Benford Analysis**: Requires large datasets (>100 samples minimum, >1000 recommended).

4. **Numerical Stability**: All KL/JS divergence calculations include epsilon (1e-10) for stability.

5. **Chaotic Systems**: Be aware that constraints can significantly alter system dynamics.

6. **Log-Periodicity**: Detection significance requires sufficient data points and clear periodic structure.

## Troubleshooting

### Import Errors
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Or install the package
pip install -e .
```

### Memory Issues
```python
# Reduce lattice size
sampler = QuantumFieldSampler(lattice_size=32)  # instead of 64

# Reduce steps
field = sampler.generate_scalar_field(steps=500)  # instead of 1000
```

### Performance Issues
- Use smaller lattice sizes for testing
- Reduce number of MCMC steps
- Use `--quick` mode for CLI
- Consider using numba JIT compilation (already enabled in some methods)

## Further Reading

- Original QLCCE monolithic implementation: `QLCCE` file
- Package structure: `pyproject.toml` and `setup.py`
- Configuration examples: `configs/` directory

For questions and issues, please open an issue on GitHub.
