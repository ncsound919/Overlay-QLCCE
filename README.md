# QLCCE - Quantum-Logarithmic Chaotic Constraint Engine

A research environment for studying quantum field theory simulations, Benford's law compliance analysis, multi-base logarithmic transformations, and chaotic systems with adaptive constraints.

## Features

- **Quantum Field Sampler**: Lattice field generation using Metropolis-Hastings algorithm, FFT analysis, correlations, vacuum statistics, and Benford compliance checks
- **Multi-Base Log Transformer**: Logarithmic transformations with multiple bases (e, 10, 2, golden ratio, etc.), iterated/super-log/chaotic modes, divergence matrices, and log-periodicity detection
- **Chaotic Constraint System**: Lorenz/Rössler/logistic map systems with boundary/quantum constraints, Lyapunov exponents, and bifurcation detection
- **Benford Quantum Analyzer**: First/nth-digit tests (chi-square, KS), KL/JS divergence, multiscale analysis, and region-level field correlations

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/ncsound919/Overlay-QLCCE.git
cd Overlay-QLCCE

# Install in development mode
pip install -e .

# Or install with pip
pip install .
```

### Dependencies

- numpy >= 1.21.0
- scipy >= 1.7.0
- numba >= 0.54.0
- sympy >= 1.9
- matplotlib >= 3.4.0
- ipywidgets >= 7.6.0
- IPython >= 7.25.0

## Quick Start

### Command Line Interface

After installation with `pip install -e .` or `pip install .`, you can use the CLI:

```bash
# Quick analysis with default settings
qlcce --quick

# Full analysis with custom steps
qlcce --full --steps 2000

# Load configuration from file
qlcce --config configs/default.json

# Export results
qlcce --full --export results.json
```

Alternatively, without installing the package, you can use:

```bash
# Run CLI module directly
python -m qlcce.cli --quick
```

### Python API

```python
from qlcce.core import (
    QuantumFieldSampler,
    MultiLogTransformer,
    ChaoticConstraintSystem,
    BenfordQuantumAnalyzer
)
import numpy as np

# Generate quantum field
field_sampler = QuantumFieldSampler(lattice_size=64, mass=0.1, coupling=1.0)
field = field_sampler.generate_scalar_field(steps=1000)

# Apply logarithmic transformations
transformer = MultiLogTransformer()
flat_field = np.abs(field.flatten())
log_results, divergence = transformer.multi_base_transform(flat_field)

# Analyze Benford's law compliance
analyzer = BenfordQuantumAnalyzer()
benford_result = analyzer.analyze_dataset(flat_field)
print(f"Compliance score: {benford_result['compliance_score']:.4f}")

# Run chaotic system with constraints
chaos = ChaoticConstraintSystem(system_type='lorenz')
chaos.add_constraint('boundary', bounds=[(-20, 20), (-30, 30), (0, 50)])
```

## Examples

See the `examples/` directory for detailed usage examples:

- `basic_usage.py` - Basic usage of all core components

## Configuration

Configuration files are JSON format. See `configs/default.json` for an example:

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

## Research Applications

QLCCE enables end-to-end research workflows for:

1. Quantum field generation and analysis
2. Chaos theory and metrics
3. Log-periodicity detection
4. Statistical compliance analysis (Benford's law)
5. Multi-scale pattern detection

## License

See LICENSE file for details.

## Citation

If you use QLCCE in your research, please cite this repository. 
