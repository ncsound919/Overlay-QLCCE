#!/usr/bin/env python3
"""
QUANTUM-LOGARITHMIC CHAOTIC CONSTRAINT ENGINE (QLCCE)
Core Engine v1.0

This tool creates a research environment that integrates:
1. Quantum field theory simulations (lattice QCD inspired)
2. Benford's law compliance testing across scales
3. Multi-base logarithmic transformations
4. Chaotic systems with adaptive constraints
5. Emergent pattern detection at boundary conditions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import sympy as sp
from sympy.physics.quantum import Operator, Commutator
import warnings
warnings.filterwarnings('ignore')

# ==================== CORE PHYSICS ENGINE ====================

class QuantumFieldSampler:
    """
    Simulates quantum field fluctuations using lattice methods
    with built-in Benford's law compliance checking
    """
    
    def __init__(self, lattice_size=64, coupling=1.0, mass=0.1, hbar=1.0):
        self.N = lattice_size
        self.g = coupling  # coupling constant
        self.m = mass      # field mass
        self.hbar = hbar
        self.lattice = None
        self.benford_deviations = []
        
        # Initialize symbolic quantum operators
        self._setup_symbolic_operators()
        
    def _setup_symbolic_operators(self):
        """Setup symbolic quantum field operators"""
        x, p = sp.symbols('x p', commutative=False)
        self.x_op = Operator('x')
        self.p_op = Operator('p')
        
        # Commutation relation [x, p] = iħ
        self.commutator = Commutator(self.x_op, self.p_op)
        
    def generate_scalar_field(self, steps=1000, temperature=1.0):
        """
        Generate quantum scalar field using Metropolis-Hastings algorithm
        on a Euclidean lattice (Wick-rotated quantum field theory)
        """
        N = self.N
        m = self.m
        g = self.g
        lattice = np.random.randn(N, N).astype(np.float32)
        
        # Action: S = ∫ d²x [½(∂φ)² + ½m²φ² + (g/4!)φ⁴]
        beta = 1.0 / temperature
        
        for step in range(steps):
            # Parallel update using checkerboard pattern
            for parity in range(2):
                for i in range(N):
                    for j in range(N):
                        if (i + j) % 2 == parity:
                            # Current action contribution
                            phi = lattice[i, j]
                            phi_nn = (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] + 
                                     lattice[i, (j+1)%N] + lattice[i, (j-1)%N]) / 4.0
                            
                            # Local action: ½(φ - φ_nn)² + ½m²φ² + (g/24)φ⁴
                            S_old = 0.5 * (phi - phi_nn)**2 + 0.5 * m**2 * phi**2 + (g / 24.0) * phi**4
                            
                            # Propose change
                            phi_new = phi + np.random.normal(0, 0.1)
                            S_new = 0.5 * (phi_new - phi_nn)**2 + 0.5 * m**2 * phi_new**2 + (g / 24.0) * phi_new**4
                            
                            # Metropolis acceptance
                            delta_S = S_new - S_old
                            if delta_S < 0 or np.random.random() < np.exp(-beta * delta_S):
                                lattice[i, j] = phi_new
            
            # Every 100 steps, check Benford's law compliance
            if step % 100 == 0:
                self._check_benford_compliance(lattice)
        
        self.lattice = lattice
        return lattice
    
    def _check_benford_compliance(self, lattice):
        """Check if field values obey Benford's law"""
        # Extract significant digits from absolute values
        flat_data = np.abs(lattice.flatten())
        # Filter out zeros and get first digits
        non_zero = flat_data[flat_data > 0]
        
        if len(non_zero) > 100:
            # Get first digits using log10 method
            first_digits = np.floor(10**(np.log10(non_zero) - np.floor(np.log10(non_zero))))
            first_digits = first_digits.astype(int)
            
            # Calculate distribution
            digit_counts = np.bincount(first_digits, minlength=10)[1:]
            observed = digit_counts / digit_counts.sum()
            
            # Benford's law theoretical distribution
            benford = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
            
            # Calculate deviation (KL divergence)
            kl_div = np.sum(observed * np.log(observed / benford))
            self.benford_deviations.append(kl_div)
            
            return kl_div
        return None
    
    def measure_correlation_function(self, max_distance=None):
        """Measure two-point correlation function <φ(x)φ(y)>"""
        if self.lattice is None:
            raise ValueError("Generate field first")
        
        N = self.N
        if max_distance is None:
            max_distance = N // 2
        
        correlations = np.zeros(max_distance)
        counts = np.zeros(max_distance)
        
        for i in range(N):
            for j in range(N):
                for d in range(1, max_distance):
                    # Average over all pairs at distance d
                    val = (self.lattice[i, j] * 
                           (self.lattice[(i+d)%N, j] + 
                            self.lattice[i, (j+d)%N] +
                            self.lattice[(i-d)%N, j] + 
                            self.lattice[i, (j-d)%N]) / 4.0)
                    correlations[d] += val
                    counts[d] += 1
        
        correlations = correlations / counts
        return correlations
    
    def fourier_transform_field(self):
        """Compute momentum-space representation"""
        if self.lattice is None:
            raise ValueError("Generate field first")
        
        # 2D FFT
        ft_field = np.fft.fft2(self.lattice)
        ft_power = np.abs(ft_field)**2
        
        # Shift zero frequency to center
        ft_power_shifted = np.fft.fftshift(ft_power)
        
        # Create momentum grid
        kx = np.fft.fftfreq(self.N, d=1.0/self.N)
        ky = np.fft.fftfreq(self.N, d=1.0/self.N)
        kx_shifted = np.fft.fftshift(kx)
        ky_shifted = np.fft.fftshift(ky)
        
        return ft_power_shifted, kx_shifted, ky_shifted
    
    def calculate_vacuum_fluctuations(self):
        """Calculate vacuum expectation values and fluctuations"""
        if self.lattice is None:
            raise ValueError("Generate field first")
        
        field_vals = self.lattice.flatten()
        
        results = {
            'mean': np.mean(field_vals),
            'variance': np.var(field_vals),
            'skewness': stats.skew(field_vals),
            'kurtosis': stats.kurtosis(field_vals),
            'entropy': self._calculate_field_entropy(field_vals),
            'benford_compliance': self.benford_deviations[-1] if self.benford_deviations else None
        }
        
        return results
    
    def _calculate_field_entropy(self, values, bins=100):
        """Calculate Shannon entropy of field distribution"""
        hist, _ = np.histogram(values, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log(hist)) / np.log(2)  # bits
        return entropy

# ==================== LOGARITHMIC TRANSFORM ENGINE ====================

class MultiLogTransformer:
    """
    Multi-base logarithmic transformations with chaotic constraint adaptation
    """
    
    def __init__(self):
        # Standard bases plus special bases
        self.bases = {
            'e': np.e,
            '10': 10,
            '2': 2,
            'golden': (1 + np.sqrt(5)) / 2,  # φ
            'silver': 1 + np.sqrt(2),        # δₛ
            'bronze': (3 + np.sqrt(13)) / 2, # σ
            'pi': np.pi,
            'sqrt2': np.sqrt(2),
            'fibonacci': self._fibonacci_base()  # Fibonacci base
        }
        
    def _fibonacci_base(self):
        """Create Fibonacci-based logarithmic system"""
        # Using φ (golden ratio) as base, but with Fibonacci scaling
        return (1 + np.sqrt(5)) / 2
    
    def transform(self, data, base='e', mode='direct'):
        """
        Apply logarithmic transformation with different modes
        
        Parameters:
        -----------
        data : array-like
            Input data (must be positive for direct log)
        base : str or float
            Logarithmic base
        mode : str
            'direct' : y = log_base(x)
            'iterated' : y = log_base(log_base(...(x)))
            'super-log' : inverse of tetration
            'chaotic' : log with chaotic perturbation
        """
        if isinstance(base, str):
            if base not in self.bases:
                raise ValueError(f"Base {base} not recognized. Available: {list(self.bases.keys())}")
            base_val = self.bases[base]
        else:
            base_val = base
        
        data = np.array(data, dtype=np.float64)
        
        if mode == 'direct':
            # Standard logarithm
            return np.log(data) / np.log(base_val)
        
        elif mode == 'iterated':
            # Iterated logarithm (log*(x))
            result = np.zeros_like(data)
            for i in range(len(data)):
                x = data[i]
                count = 0
                while x >= base_val:
                    x = np.log(x) / np.log(base_val)
                    count += 1
                result[i] = count + x
            return result
        
        elif mode == 'super-log':
            # Super-logarithm (inverse of tetration)
            result = np.zeros_like(data)
            for i in range(len(data)):
                x = data[i]
                # Approximate using Lambert W function
                if x <= 0:
                    result[i] = -np.inf
                else:
                    # slog_b(x) ≈ log_b(log_b(x)) + 1 for large x
                    if x > base_val:
                        result[i] = np.log(np.log(x) / np.log(base_val)) / np.log(base_val) + 1
                    else:
                        result[i] = np.log(x) / np.log(base_val)
            return result
        
        elif mode == 'chaotic':
            # Logarithm with chaotic perturbation
            log_result = np.log(data) / np.log(base_val)
            # Add Lorenz system perturbation
            chaotic_perturbation = self._lorenz_perturbation(len(data))
            return log_result * (1 + 0.1 * chaotic_perturbation)
        
        else:
            raise ValueError(f"Mode {mode} not recognized")
    
    def _lorenz_perturbation(self, n_points, sigma=10, rho=28, beta=8/3, dt=0.01):
        """Generate chaotic perturbation using Lorenz system"""
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        z = np.zeros(n_points)
        
        # Initial conditions
        x[0], y[0], z[0] = 0.1, 0.0, 0.0
        
        for i in range(1, n_points):
            dx = sigma * (y[i-1] - x[i-1])
            dy = x[i-1] * (rho - z[i-1]) - y[i-1]
            dz = x[i-1] * y[i-1] - beta * z[i-1]
            
            x[i] = x[i-1] + dx * dt
            y[i] = y[i-1] + dy * dt
            z[i] = z[i-1] + dz * dt
        
        return x  # Return x-component as perturbation
    
    def multi_base_transform(self, data, bases=['e', '10', '2', 'golden']):
        """Apply multiple logarithmic transforms and compare"""
        results = {}
        for base in bases:
            results[base] = self.transform(data, base)
        
        # Calculate information divergence between transforms
        divergence_matrix = self._calculate_divergence_matrix(results)
        
        return results, divergence_matrix
    
    def _calculate_divergence_matrix(self, transformed_data):
        """Calculate KL divergence between different transforms"""
        bases = list(transformed_data.keys())
        n = len(bases)
        divergence = np.zeros((n, n))
        
        for i, base_i in enumerate(bases):
            for j, base_j in enumerate(bases):
                if i != j:
                    # Normalize to probability distributions
                    pi = np.abs(transformed_data[base_i])
                    pj = np.abs(transformed_data[base_j])
                    
                    pi = pi / (pi.sum() + 1e-10)
                    pj = pj / (pj.sum() + 1e-10)
                    
                    # Calculate symmetric KL divergence
                    kl_ij = np.sum(pi * np.log(pi / (pj + 1e-10) + 1e-10))
                    kl_ji = np.sum(pj * np.log(pj / (pi + 1e-10) + 1e-10))
                    
                    divergence[i, j] = (kl_ij + kl_ji) / 2
        
        return divergence
    
    def detect_log_periodicity(self, data, base='e', significance=0.01):
        """
        Detect log-periodic patterns (signatures of discrete scale invariance)
        
        Returns:
        --------
        periods : list
            Detected log-periodicities
        significance : float
            Statistical significance
        """
        log_data = self.transform(data, base)
        
        # Fourier transform of log-transformed data
        n = len(log_data)
        freq = np.fft.fftfreq(n)
        power = np.abs(np.fft.fft(log_data))**2
        
        # Find peaks in power spectrum
        peaks = self._find_peaks(power[:n//2], threshold=0.1)
        
        # Convert to log-periods
        periods = []
        for peak_idx in peaks:
            if freq[peak_idx] > 0:
                period = 1 / freq[peak_idx]
                periods.append(period)
        
        # Calculate significance using surrogate data
        sig = self._calculate_significance(log_data, periods, n_surrogates=1000)
        
        return periods, sig
    
    def _find_peaks(self, data, threshold):
        """Find peaks in 1D data"""
        peaks = []
        for i in range(1, len(data)-1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold:
                peaks.append(i)
        return peaks
    
    def _calculate_significance(self, data, periods, n_surrogates=1000):
        """Calculate significance using phase randomization"""
        # Generate surrogate data by randomizing phases
        original_power = np.abs(np.fft.fft(data))**2
        
        surrogate_peaks = []
        for _ in range(n_surrogates):
            # Randomize phases
            ft = np.fft.fft(data)
            phases = np.random.uniform(0, 2*np.pi, len(data))
            ft_surrogate = np.abs(ft) * np.exp(1j * phases)
            surrogate = np.real(np.fft.ifft(ft_surrogate))
            
            # Calculate power spectrum
            power = np.abs(np.fft.fft(surrogate))**2
            peaks = self._find_peaks(power[:len(data)//2], threshold=0.1)
            surrogate_peaks.append(len(peaks))
        
        # Compare with original
        original_n_peaks = len(periods)
        surrogate_peaks = np.array(surrogate_peaks)
        
        # p-value: probability of getting at least as many peaks by chance
        p_value = np.sum(surrogate_peaks >= original_n_peaks) / n_surrogates
        
        return 1 - p_value

# ==================== CHAOTIC CONSTRAINT SYSTEM ====================

class ChaoticConstraintSystem:
    """
    Chaotic systems with adaptive constraints and boundary conditions
    """
    
    def __init__(self, system_type='lorenz'):
        self.system_type = system_type
        self.constraints = []
        self.bifurcation_points = []
        
    def lorenz_system(self, params=None, constraints=None):
        """Lorenz system with optional constraints"""
        if params is None:
            params = {'sigma': 10, 'rho': 28, 'beta': 8/3}
        
        sigma, rho, beta = params['sigma'], params['rho'], params['beta']
        
        def equations(state, t):
            x, y, z = state
            
            # Apply constraints if any
            if constraints:
                for constraint in constraints:
                    x, y, z = constraint(state, t)
            
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            return [dx, dy, dz]
        
        return equations
    
    def rossler_system(self, params=None, constraints=None):
        """Rössler system with optional constraints"""
        if params is None:
            params = {'a': 0.2, 'b': 0.2, 'c': 5.7}
        
        a, b, c = params['a'], params['b'], params['c']
        
        def equations(state, t):
            x, y, z = state
            
            # Apply constraints
            if constraints:
                for constraint in constraints:
                    x, y, z = constraint(state, t)
            
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            
            return [dx, dy, dz]
        
        return equations
    
    def logistic_map_with_constraints(self, r=3.9, n_iter=1000, constraints=None):
        """
        Logistic map: x_{n+1} = r * x_n * (1 - x_n)
        with adaptive constraints
        """
        x = np.zeros(n_iter)
        x[0] = 0.1
        
        for i in range(1, n_iter):
            # Standard logistic map
            x_next = r * x[i-1] * (1 - x[i-1])
            
            # Apply constraints
            if constraints:
                for constraint in constraints:
                    x_next = constraint(x_next, x[i-1], i)
            
            x[i] = x_next
        
        return x
    
    def add_constraint(self, constraint_type, **kwargs):
        """Add a constraint to the system"""
        if constraint_type == 'boundary':
            constraint = self._create_boundary_constraint(**kwargs)
        elif constraint_type == 'symmetry':
            constraint = self._create_symmetry_constraint(**kwargs)
        elif constraint_type == 'conservation':
            constraint = self._create_conservation_constraint(**kwargs)
        elif constraint_type == 'quantum':
            constraint = self._create_quantum_constraint(**kwargs)
        else:
            raise ValueError(f"Constraint type {constraint_type} not recognized")
        
        self.constraints.append(constraint)
        return constraint
    
    def _create_boundary_constraint(self, bounds):
        """Create boundary constraint (reflecting/absorbing)"""
        def constraint(state, t=None):
            if isinstance(state, (list, np.ndarray)):
                # For ODE systems
                constrained_state = []
                for i, s in enumerate(state):
                    if i < len(bounds):
                        low, high = bounds[i]
                        if s < low:
                            s = low + (low - s)  # Reflect
                        elif s > high:
                            s = high - (s - high)  # Reflect
                    constrained_state.append(s)
                return constrained_state
            else:
                # For single value (logistic map)
                s = state
                if 'bounds' in bounds:
                    low, high = bounds['bounds']
                    if s < low:
                        s = low + (low - s)
                    elif s > high:
                        s = high - (s - high)
                return s
        
        return constraint
    
    def _create_symmetry_constraint(self, axis='z'):
        """Create symmetry constraint (e.g., z → -z)"""
        def constraint(state, t=None):
            if isinstance(state, (list, np.ndarray)):
                if axis == 'x':
                    state[0] = -state[0]
                elif axis == 'y':
                    state[1] = -state[1]
                elif axis == 'z':
                    state[2] = -state[2]
            return state
        return constraint
    
    def _create_conservation_constraint(self, conserved_quantity='energy'):
        """Create conservation law constraint"""
        def constraint(state, t=None):
            # Placeholder - would enforce conservation of specified quantity
            return state
        return constraint
    
    def _create_quantum_constraint(self, uncertainty=0.1):
        """Create quantum uncertainty constraint"""
        def constraint(state, t=None):
            if isinstance(state, (list, np.ndarray)):
                # Add Heisenberg-like uncertainty
                constrained_state = []
                for s in state:
                    # Add uncertainty proportional to current value
                    s = s + np.random.normal(0, uncertainty * abs(s))
                    constrained_state.append(s)
                return constrained_state
            else:
                return state + np.random.normal(0, uncertainty * abs(state))
        return constraint
    
    def calculate_lyapunov_exponents(self, trajectory, dt=0.01):
        """
        Calculate Lyapunov exponents from trajectory data
        """
        n = len(trajectory)
        if n < 100:
            return [0]
        
        # Reconstruct phase space using time-delay embedding
        embedded = self._time_delay_embedding(trajectory)
        
        # Calculate Lyapunov exponents using Rosenstein's algorithm
        m = embedded.shape[1]
        lexp = np.zeros(m)
        
        # For simplicity, use a basic estimation
        # In practice, use more robust algorithm
        for i in range(m):
            # Calculate divergence of nearby trajectories
            divergences = []
            for j in range(10, n-10):
                ref_point = embedded[j, i]
                # Find nearest neighbor
                distances = np.abs(embedded[:, i] - ref_point)
                distances[j-5:j+5] = np.inf  # Exclude immediate neighbors
                nn_idx = np.argmin(distances)
                
                divergence = np.abs(embedded[j+1, i] - embedded[nn_idx+1, i])
                initial_dist = np.abs(embedded[j, i] - embedded[nn_idx, i])
                
                if initial_dist > 0:
                    divergences.append(np.log(divergence / initial_dist))
            
            if divergences:
                lexp[i] = np.mean(divergences) / dt
        
        return lexp
    
    def _time_delay_embedding(self, signal, delay=1, dimension=3):
        """Create time-delay embedding of signal"""
        n = len(signal)
        embedded = np.zeros((n - (dimension-1)*delay, dimension))
        
        for i in range(dimension):
            embedded[:, i] = signal[i*delay:i*delay + embedded.shape[0]]
        
        return embedded
    
    def detect_bifurcations(self, parameter_range, n_points=100):
        """
        Detect bifurcation points in parameter space
        """
        bifurcations = []
        
        if self.system_type == 'logistic':
            r_values = np.linspace(parameter_range[0], parameter_range[1], n_points)
            
            for r in r_values:
                # Generate trajectory
                trajectory = self.logistic_map_with_constraints(r=r, n_iter=1000)
                
                # Analyze last 100 points for periodicity
                final_states = trajectory[-100:]
                unique_states = np.unique(np.round(final_states, 4))
                
                # Record number of unique states (period)
                period = len(unique_states)
                
                # Check if period changes
                if len(bifurcations) > 0:
                    if period != bifurcations[-1]['period']:
                        bifurcations.append({
                            'parameter': r,
                            'period': period,
                            'type': 'period_doubling' if period > bifurcations[-1]['period'] else 'other'
                        })
                else:
                    bifurcations.append({
                        'parameter': r,
                        'period': period,
                        'type': 'initial'
                    })
        
        return bifurcations

# ==================== BENFORD'S LAW ANALYZER ====================

class BenfordQuantumAnalyzer:
    """
    Advanced Benford's law analysis with quantum field correlations
    """
    
    def __init__(self):
        self.expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
    
    def analyze_dataset(self, data, digit_position=1):
        """
        Analyze dataset for Benford's law compliance
        
        Parameters:
        -----------
        data : array-like
            Input data (positive values)
        digit_position : int
            1 for first digit, 2 for second digit, etc.
        """
        data = np.array(data)
        data = data[data > 0]  # Remove non-positive values
        
        if len(data) == 0:
            return None
        
        # Extract digits
        if digit_position == 1:
            digits = self._extract_first_digits(data)
        else:
            digits = self._extract_nth_digit(data, digit_position)
        
        # Calculate observed distribution
        digit_counts = np.bincount(digits, minlength=10)[1:] if digit_position == 1 else np.bincount(digits, minlength=10)
        total = digit_counts.sum()
        
        if total == 0:
            return None
        
        observed = digit_counts / total
        
        # Statistical tests
        chi2, p_value_chi2 = self._chi_square_test(digit_counts, total, digit_position)
        ks_stat, p_value_ks = self._ks_test(observed, digit_position)
        
        # Calculate divergence measures
        kl_div = self._kl_divergence(observed, digit_position)
        js_div = self._js_divergence(observed, digit_position)
        
        results = {
            'observed_distribution': observed,
            'expected_distribution': self.expected if digit_position == 1 else self._expected_nth_digit(digit_position),
            'chi2_statistic': chi2,
            'chi2_p_value': p_value_chi2,
            'ks_statistic': ks_stat,
            'ks_p_value': p_value_ks,
            'kl_divergence': kl_div,
            'js_divergence': js_div,
            'compliance_score': 1 - js_div,  # Higher = more compliant
            'digit_frequencies': digit_counts,
            'total_samples': total
        }
        
        return results
    
    def _extract_first_digits(self, data):
        """Extract first significant digit using log10 method"""
        first_digits = np.floor(10**(np.log10(data) - np.floor(np.log10(data))))
        return first_digits.astype(int)
    
    def _extract_nth_digit(self, data, n):
        """Extract nth digit (n=1 for first, n=2 for second, etc.)"""
        # Convert to string representation and extract digit
        digits = []
        for x in data:
            s = f"{x:.10f}".replace('.', '').lstrip('0')
            if len(s) >= n:
                digits.append(int(s[n-1]))
        return np.array(digits)
    
    def _expected_nth_digit(self, n):
        """Calculate expected distribution for nth digit"""
        # For n > 1, distribution is more uniform
        if n == 1:
            return self.expected
        else:
            # Approximate distribution for nth digit
            # Exact formula: P(d) = Σ_{k=10^{n-2}}^{10^{n-1}-1} log10(1 + 1/(10k + d))
            probs = np.zeros(10)
            for d in range(10):
                prob = 0
                for k in range(10**(n-2), 10**(n-1)):
                    prob += np.log10(1 + 1/(10*k + d))
                probs[d] = prob
            return probs / probs.sum()
    
    def _chi_square_test(self, observed_counts, total, digit_position):
        """Chi-square goodness-of-fit test"""
        expected_counts = self.expected * total if digit_position == 1 else self._expected_nth_digit(digit_position) * total
        expected_counts = expected_counts[expected_counts > 0]
        observed_counts = observed_counts[expected_counts > 0]
        
        chi2 = np.sum((observed_counts - expected_counts)**2 / expected_counts)
        df = len(expected_counts) - 1
        p_value = 1 - stats.chi2.cdf(chi2, df)
        
        return chi2, p_value
    
    def _ks_test(self, observed, digit_position):
        """Kolmogorov-Smirnov test"""
        expected_cdf = np.cumsum(self.expected if digit_position == 1 else self._expected_nth_digit(digit_position))
        observed_cdf = np.cumsum(observed)
        
        ks_stat = np.max(np.abs(observed_cdf - expected_cdf))
        
        # Approximate p-value
        n = len(observed)
        p_value = stats.kstwo.sf(ks_stat, n)
        
        return ks_stat, p_value
    
    def _kl_divergence(self, observed, digit_position):
        """Kullback-Leibler divergence"""
        expected = self.expected if digit_position == 1 else self._expected_nth_digit(digit_position)
        # Add small epsilon to avoid log(0)
        kl = np.sum(observed * np.log(observed / (expected + 1e-10) + 1e-10))
        return kl
    
    def _kl_divergence_general(self, p, q):
        """Kullback-Leibler divergence between two arbitrary discrete distributions p and q."""
        eps = 1e-10
        p_safe = p + eps
        q_safe = q + eps
        return np.sum(p_safe * np.log(p_safe / q_safe))
    
    def _js_divergence(self, observed, digit_position):
        """Jensen-Shannon divergence (symmetric)"""
        expected = self.expected if digit_position == 1 else self._expected_nth_digit(digit_position)
        m = 0.5 * (observed + expected)
        js = 0.5 * self._kl_divergence_general(observed, m) + 0.5 * self._kl_divergence_general(expected, m)
        return js
    
    def analyze_multiscale(self, data, scales=None):
        """
        Analyze Benford compliance at multiple scales (logarithmic bins)
        """
        if scales is None:
            scales = [1, 10, 100, 1000]
        
        results = {}
        data = np.array(data)
        
        for scale in scales:
            # Bin data logarithmically
            if scale > 1:
                bins = np.logspace(np.log10(data.min()), np.log10(data.max()), scale)
                binned_results = []
                
                for i in range(len(bins)-1):
                    mask = (data >= bins[i]) & (data < bins[i+1])
                    binned_data = data[mask]
                    
                    if len(binned_data) > 10:  # Require minimum samples
                        benford_result = self.analyze_dataset(binned_data)
                        if benford_result:
                            binned_results.append(benford_result)
                
                if binned_results:
                    # Average compliance across bins
                    avg_compliance = np.mean([r['compliance_score'] for r in binned_results])
                    results[f'scale_{scale}'] = {
                        'avg_compliance': avg_compliance,
                        'n_bins': len(binned_results),
                        'binned_results': binned_results
                    }
        
        return results
    
    def quantum_benford_correlation(self, field_data, position_correlation=False):
        """
        Analyze correlation between Benford compliance and quantum field properties
        """
        if isinstance(field_data, dict) and 'lattice' in field_data:
            lattice = field_data['lattice']
        else:
            lattice = field_data
        
        # Analyze Benford compliance in different regions
        region_size = lattice.shape[0] // 2
        
        region_compliance = []
        region_properties = []
        
        for i in range(2):
            for j in range(2):
                region = lattice[i*region_size:(i+1)*region_size, j*region_size:(j+1)*region_size]
                
                # Benford analysis
                flat_region = np.abs(region.flatten())
                benford_result = self.analyze_dataset(flat_region)
                
                if benford_result:
                    region_compliance.append(benford_result['compliance_score'])
                    
                    # Region properties
                    region_properties.append({
                        'mean': np.mean(region),
                        'variance': np.var(region),
                        'entropy': stats.entropy(np.histogram(region, bins=50, density=True)[0])
                    })
        
        # Calculate correlations
        if len(region_compliance) > 1:
            compliance_array = np.array(region_compliance)
            properties_array = np.array([list(p.values()) for p in region_properties])
            
            # Correlation matrix - ensure proper shape
            try:
                combined = np.column_stack([compliance_array.reshape(-1, 1), properties_array])
                correlation_matrix = np.corrcoef(combined.T)
            except Exception:
                correlation_matrix = np.array([[1.0]])
            
            return {
                'region_compliance': region_compliance,
                'region_properties': region_properties,
                'correlation_matrix': correlation_matrix,
                'mean_compliance': np.mean(region_compliance),
                'std_compliance': np.std(region_compliance)
            }
        
        return None

# ==================== MAIN INTEGRATION ENGINE ====================

class QLCCE_Engine:
    """
    Main integration engine for Quantum-Logarithmic Chaotic Constraint Engine
    """
    
    def __init__(self, config=None):
        if config is None:
            config = {
                'lattice_size': 64,
                'field_mass': 0.1,
                'field_coupling': 1.0,
                'chaos_system': 'lorenz',
                'log_bases': ['e', '10', '2', 'golden'],
                'constraints': ['boundary', 'quantum']
            }
        
        self.config = config
        
        # Initialize components
        self.quantum_field = QuantumFieldSampler(
            lattice_size=config['lattice_size'],
            mass=config['field_mass'],
            coupling=config['field_coupling']
        )
        
        self.log_transformer = MultiLogTransformer()
        self.chaos_system = ChaoticConstraintSystem(system_type=config['chaos_system'])
        self.benford_analyzer = BenfordQuantumAnalyzer()
        
        # Results storage
        self.results = {}
        
    def run_full_analysis(self, n_steps=1000):
        """
        Run complete analysis pipeline
        """
        print("=" * 60)
        print("QLCCE: Starting full analysis")
        print("=" * 60)
        
        # Step 1: Generate quantum field
        print("\n1. Generating quantum scalar field...")
        field = self.quantum_field.generate_scalar_field(steps=n_steps)
        field_properties = self.quantum_field.calculate_vacuum_fluctuations()
        
        self.results['field'] = field
        self.results['field_properties'] = field_properties
        print(f"   Field generated: {field.shape}")
        print(f"   Vacuum fluctuations: {field_properties}")
        
        # Step 2: Apply logarithmic transformations
        print("\n2. Applying multi-base logarithmic transforms...")
        flat_field = np.abs(field.flatten())
        log_results, divergence_matrix = self.log_transformer.multi_base_transform(
            flat_field, 
            bases=self.config['log_bases']
        )
        
        self.results['log_transforms'] = log_results
        self.results['log_divergence'] = divergence_matrix
        
        # Find optimal base (minimum divergence from others)
        optimal_base = self._find_optimal_base(divergence_matrix)
        print(f"   Optimal logarithmic base: {self.config['log_bases'][optimal_base]}")
        
        # Step 3: Benford's law analysis
        print("\n3. Analyzing Benford's law compliance...")
        benford_result = self.benford_analyzer.analyze_dataset(flat_field)
        
        if benford_result:
            self.results['benford'] = benford_result
            print(f"   Benford compliance score: {benford_result['compliance_score']:.4f}")
            print(f"   Chi-square p-value: {benford_result['chi2_p_value']:.6f}")
        
        # Step 4: Multiscale Benford analysis
        print("\n4. Multiscale Benford analysis...")
        multiscale_results = self.benford_analyzer.analyze_multiscale(flat_field)
        self.results['multiscale_benford'] = multiscale_results
        
        # Step 5: Quantum-Benford correlation
        print("\n5. Quantum-Benford correlation analysis...")
        quantum_benford_corr = self.benford_analyzer.quantum_benford_correlation(field)
        if quantum_benford_corr:
            self.results['quantum_benford_correlation'] = quantum_benford_corr
            print(f"   Mean region compliance: {quantum_benford_corr['mean_compliance']:.4f}")
        
        # Step 6: Chaotic system with constraints
        print("\n6. Running chaotic system with constraints...")
        
        # Add constraints
        constraints = []
        for constraint_type in self.config.get('constraints', []):
            if constraint_type == 'boundary':
                constraint = self.chaos_system.add_constraint('boundary', 
                                                             bounds=[(-20, 20), (-30, 30), (0, 50)])
            elif constraint_type == 'quantum':
                constraint = self.chaos_system.add_constraint('quantum', uncertainty=0.05)
            constraints.append(constraint)
        
        # Generate chaotic trajectory
        if self.config['chaos_system'] == 'lorenz':
            equations = self.chaos_system.lorenz_system(constraints=constraints)
            
            # Integrate
            from scipy.integrate import odeint
            t = np.linspace(0, 100, 10000)
            initial_state = [0.1, 0.0, 0.0]
            trajectory = odeint(equations, initial_state, t)
            
        elif self.config['chaos_system'] == 'logistic':
            trajectory = self.chaos_system.logistic_map_with_constraints(
                r=3.9, 
                n_iter=10000,
                constraints=constraints
            )
            trajectory = trajectory.reshape(-1, 1)
        
        self.results['chaos_trajectory'] = trajectory
        
        # Calculate Lyapunov exponents
        print("7. Calculating Lyapunov exponents...")
        if trajectory.shape[1] > 1:
            lyapunovs = self.chaos_system.calculate_lyapunov_exponents(trajectory[:, 0])
        else:
            lyapunovs = self.chaos_system.calculate_lyapunov_exponents(trajectory.flatten())
        
        self.results['lyapunov_exponents'] = lyapunovs
        print(f"   Lyapunov exponents: {lyapunovs}")
        
        # Step 8: Log-periodicity detection
        print("\n8. Detecting log-periodic patterns...")
        optimal_base_name = self.config['log_bases'][optimal_base]
        log_periods, significance = self.log_transformer.detect_log_periodicity(
            flat_field[:1000],  # Use subset for speed
            base=optimal_base_name
        )
        
        self.results['log_periodicity'] = {
            'periods': log_periods,
            'significance': significance,
            'base': optimal_base_name
        }
        
        if log_periods:
            print(f"   Detected {len(log_periods)} log-periodicities")
            print(f"   Significance: {significance:.4f}")
        
        print("\n" + "=" * 60)
        print("QLCCE: Analysis complete")
        print("=" * 60)
        
        return self.results
    
    def _find_optimal_base(self, divergence_matrix):
        """Find the logarithmic base with minimum average divergence from others"""
        # Average divergence for each base
        avg_divergence = np.mean(divergence_matrix, axis=1)
        # Ignore diagonal (self-divergence, which is 0)
        for i in range(len(avg_divergence)):
            avg_divergence[i] = np.mean([divergence_matrix[i, j] for j in range(len(avg_divergence)) if i != j])
        
        optimal_idx = np.argmin(avg_divergence)
        return optimal_idx
    
    def visualize_results(self, save_figures=False):
        """Create comprehensive visualization of results"""
        if not self.results:
            print("No results to visualize. Run analysis first.")
            return
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Quantum field visualization
        ax1 = fig.add_subplot(3, 4, 1)
        if 'field' in self.results:
            field = self.results['field']
            im = ax1.imshow(field, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im, ax=ax1, shrink=0.7)
            ax1.set_title('Quantum Scalar Field')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
        
        # 2. Field correlation function
        ax2 = fig.add_subplot(3, 4, 2)
        if 'field' in self.results:
            correlations = self.quantum_field.measure_correlation_function()
            ax2.plot(correlations[1:], 'b-', linewidth=2)
            ax2.set_title('Two-Point Correlation Function')
            ax2.set_xlabel('Distance')
            ax2.set_ylabel('<φ(x)φ(y)>')
            ax2.grid(True, alpha=0.3)
        
        # 3. Benford's law compliance
        ax3 = fig.add_subplot(3, 4, 3)
        if 'benford' in self.results:
            benford = self.results['benford']
            digits = np.arange(1, 10)
            ax3.bar(digits - 0.2, benford['observed_distribution'], width=0.4, 
                   label='Observed', alpha=0.7)
            ax3.bar(digits + 0.2, benford['expected_distribution'], width=0.4, 
                   label='Expected', alpha=0.7)
            ax3.set_title(f"Benford's Law (compliance: {benford['compliance_score']:.3f})")
            ax3.set_xlabel('First Digit')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Logarithmic transform comparison
        ax4 = fig.add_subplot(3, 4, 4)
        if 'log_transforms' in self.results:
            log_results = self.results['log_transforms']
            for i, (base, values) in enumerate(log_results.items()):
                if i < 4:  # Plot only first 4
                    ax4.hist(values[:1000], bins=50, alpha=0.5, label=base, density=True)
            ax4.set_title('Logarithmic Transform Distributions')
            ax4.set_xlabel('Transformed Value')
            ax4.set_ylabel('Density')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Chaotic trajectory
        ax5 = fig.add_subplot(3, 4, 5, projection='3d')
        if 'chaos_trajectory' in self.results:
            trajectory = self.results['chaos_trajectory']
            if trajectory.shape[1] >= 3:
                ax5.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                        'b-', alpha=0.6, linewidth=0.5)
                ax5.set_title('Chaotic Trajectory (3D)')
                ax5.set_xlabel('X')
                ax5.set_ylabel('Y')
                ax5.set_zlabel('Z')
            elif trajectory.shape[1] == 2:
                ax5 = fig.add_subplot(3, 4, 5)
                ax5.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.6, linewidth=0.5)
                ax5.set_title('Chaotic Trajectory (2D)')
                ax5.set_xlabel('X')
                ax5.set_ylabel('Y')
                ax5.grid(True, alpha=0.3)
            else:
                ax5 = fig.add_subplot(3, 4, 5)
                ax5.plot(trajectory[:1000], 'b-', alpha=0.6, linewidth=1)
                ax5.set_title('Chaotic Trajectory (1D)')
                ax5.set_xlabel('Time')
                ax5.set_ylabel('Value')
                ax5.grid(True, alpha=0.3)
        
        # 6. Lyapunov spectrum
        ax6 = fig.add_subplot(3, 4, 6)
        if 'lyapunov_exponents' in self.results:
            lexp = self.results['lyapunov_exponents']
            ax6.bar(range(len(lexp)), lexp, color='green', alpha=0.7)
            ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax6.set_title('Lyapunov Exponents')
            ax6.set_xlabel('Exponent Index')
            ax6.set_ylabel('Value')
            ax6.grid(True, alpha=0.3)
        
        # 7. Fourier transform of field
        ax7 = fig.add_subplot(3, 4, 7)
        if 'field' in self.results:
            field = self.results['field']
            ft_power, kx, ky = self.quantum_field.fourier_transform_field()
            # Plot radial average
            k_mag = np.sqrt(kx[:, np.newaxis]**2 + ky**2)
            k_flat = k_mag.flatten()
            power_flat = ft_power.flatten()
            
            # Bin by k magnitude
            k_bins = np.linspace(0, k_flat.max(), 50)
            bin_centers = (k_bins[:-1] + k_bins[1:]) / 2
            bin_powers = np.zeros(len(bin_centers))
            
            for i in range(len(bin_centers)):
                mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
                if np.any(mask):
                    bin_powers[i] = np.mean(power_flat[mask])
            
            ax7.loglog(bin_centers[1:], bin_powers[1:], 'r-', linewidth=2)
            ax7.set_title('Power Spectrum (Radial Average)')
            ax7.set_xlabel('|k|')
            ax7.set_ylabel('Power')
            ax7.grid(True, alpha=0.3, which='both')
        
        # 8. Log-periodicity
        ax8 = fig.add_subplot(3, 4, 8)
        if 'log_periodicity' in self.results:
            log_per = self.results['log_periodicity']
            if log_per['periods']:
                periods = log_per['periods']
                ax8.bar(range(len(periods)), periods, color='purple', alpha=0.7)
                ax8.set_title(f"Log-Periodicities (base: {log_per['base']})")
                ax8.set_xlabel('Period Index')
                ax8.set_ylabel('Period')
                ax8.grid(True, alpha=0.3)
        
        # 9. Divergence matrix
        ax9 = fig.add_subplot(3, 4, 9)
        if 'log_divergence' in self.results:
            divergence = self.results['log_divergence']
            im = ax9.imshow(divergence, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax9, shrink=0.7)
            ax9.set_title('Log Transform Divergence Matrix')
            ax9.set_xlabel('Base Index')
            ax9.set_ylabel('Base Index')
            # Add labels
            if 'log_bases' in self.config:
                bases = self.config['log_bases'][:len(divergence)]
                ax9.set_xticks(range(len(bases)))
                ax9.set_yticks(range(len(bases)))
                ax9.set_xticklabels(bases, rotation=45)
                ax9.set_yticklabels(bases)
        
        # 10. Quantum-Benford correlation
        ax10 = fig.add_subplot(3, 4, 10)
        if 'quantum_benford_correlation' in self.results:
            corr = self.results['quantum_benford_correlation']['correlation_matrix']
            im = ax10.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax10, shrink=0.7)
            ax10.set_title('Quantum-Benford Correlation')
            ax10.set_xlabel('Variable')
            ax10.set_ylabel('Variable')
        
        # 11. Multiscale Benford
        ax11 = fig.add_subplot(3, 4, 11)
        if 'multiscale_benford' in self.results:
            multiscale = self.results['multiscale_benford']
            scales = []
            compliances = []
            for scale, data in multiscale.items():
                scales.append(int(scale.split('_')[1]))
                compliances.append(data['avg_compliance'])
            
            ax11.semilogx(scales, compliances, 'go-', linewidth=2, markersize=8)
            ax11.set_title('Benford Compliance vs Scale')
            ax11.set_xlabel('Number of Bins (log scale)')
            ax11.set_ylabel('Average Compliance')
            ax11.grid(True, alpha=0.3, which='both')
        
        # 12. Field histogram with log-normal fit
        ax12 = fig.add_subplot(3, 4, 12)
        if 'field' in self.results:
            field_flat = np.abs(self.results['field'].flatten())
            field_flat = field_flat[field_flat > 0]
            
            # Histogram on log scale
            log_field = np.log(field_flat)
            ax12.hist(log_field, bins=50, density=True, alpha=0.7, 
                     label='Field values (log)')
            
            # Fit normal distribution
            mu, sigma = np.mean(log_field), np.std(log_field)
            x = np.linspace(log_field.min(), log_field.max(), 100)
            pdf = stats.norm.pdf(x, mu, sigma)
            ax12.plot(x, pdf, 'r-', linewidth=2, 
                     label=f'Normal fit: μ={mu:.2f}, σ={sigma:.2f}')
            
            ax12.set_title('Field Distribution (Log Scale)')
            ax12.set_xlabel('log(φ)')
            ax12.set_ylabel('Density')
            ax12.legend()
            ax12.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_figures:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'qlcce_results_{timestamp}.png', dpi=150, bbox_inches='tight')
            print(f"Results saved as qlcce_results_{timestamp}.png")
        
        plt.show()
    
    def export_results(self, filename=None):
        """Export results to JSON file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        exportable_results = convert_for_json(self.results)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'qlcce_export_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(exportable_results, f, indent=2)
        
        print(f"Results exported to {filename}")
        return filename

# ==================== INTERACTIVE RESEARCH NOTEBOOK ====================

def create_research_notebook():
    """
    Create an interactive Jupyter notebook-like interface
    for exploring QLCCE concepts
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    
    # Only run in Jupyter environment
    try:
        get_ipython()
    except NameError:
        print("This function requires Jupyter/IPython environment")
        return
    
    # Create widgets
    lattice_size_slider = widgets.IntSlider(
        value=32, min=16, max=128, step=16,
        description='Lattice Size:',
        continuous_update=False
    )
    
    mass_slider = widgets.FloatSlider(
        value=0.1, min=0.01, max=1.0, step=0.01,
        description='Field Mass:',
        continuous_update=False
    )
    
    coupling_slider = widgets.FloatSlider(
        value=1.0, min=0.1, max=5.0, step=0.1,
        description='Coupling:',
        continuous_update=False
    )
    
    chaos_system_dropdown = widgets.Dropdown(
        options=['lorenz', 'rossler', 'logistic'],
        value='lorenz',
        description='Chaos System:'
    )
    
    log_bases_select = widgets.SelectMultiple(
        options=['e', '10', '2', 'golden', 'silver', 'bronze', 'pi', 'sqrt2'],
        value=('e', '10', '2'),
        description='Log Bases:',
        rows=4
    )
    
    constraints_select = widgets.SelectMultiple(
        options=['none', 'boundary', 'symmetry', 'conservation', 'quantum'],
        value=('boundary', 'quantum'),
        description='Constraints:',
        rows=3
    )
    
    run_button = widgets.Button(
        description='Run Analysis',
        button_style='success',
        tooltip='Run the QLCCE analysis'
    )
    
    output = widgets.Output()
    
    def on_run_button_clicked(b):
        with output:
            clear_output()
            
            # Create configuration
            config = {
                'lattice_size': lattice_size_slider.value,
                'field_mass': mass_slider.value,
                'field_coupling': coupling_slider.value,
                'chaos_system': chaos_system_dropdown.value,
                'log_bases': list(log_bases_select.value),
                'constraints': list(constraints_select.value)
            }
            
            # Remove 'none' if present
            if 'none' in config['constraints']:
                config['constraints'] = []
            
            # Initialize and run engine
            engine = QLCCE_Engine(config)
            results = engine.run_full_analysis(n_steps=500)
            
            # Visualize
            engine.visualize_results(save_figures=False)
            
            # Show key results
            print("\n" + "="*60)
            print("KEY FINDINGS:")
            print("="*60)
            
            if 'benford' in results:
                benford = results['benford']
                print(f"• Benford Compliance: {benford['compliance_score']:.4f}")
                print(f"• Chi-square p-value: {benford['chi2_p_value']:.6f}")
            
            if 'lyapunov_exponents' in results:
                lexp = results['lyapunov_exponents']
                positive_lexp = [l for l in lexp if l > 0]
                print(f"• Positive Lyapunov exponents: {len(positive_lexp)}")
                if positive_lexp:
                    print(f"• Largest Lyapunov: {max(positive_lexp):.4f}")
            
            if 'log_periodicity' in results:
                log_per = results['log_periodicity']
                if log_per['periods']:
                    print(f"• Log-periodicities detected: {len(log_per['periods'])}")
                    print(f"• Significance: {log_per['significance']:.4f}")
            
            if 'field_properties' in results:
                field_props = results['field_properties']
                print(f"• Field entropy: {field_props['entropy']:.4f} bits")
                if field_props['benford_compliance']:
                    print(f"• Benford deviation: {field_props['benford_compliance']:.4f}")
    
    run_button.on_click(on_run_button_clicked)
    
    # Display interface
    display(widgets.VBox([
        widgets.HTML("<h2>QLCCE Interactive Research Interface</h2>"),
        widgets.HBox([lattice_size_slider, mass_slider, coupling_slider]),
        widgets.HBox([chaos_system_dropdown]),
        widgets.HBox([log_bases_select, constraints_select]),
        run_button,
        output
    ]))

# ==================== COMMAND LINE INTERFACE ====================

def main():
    """Command line interface for QLCCE"""
    import argparse
    
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
    parser.add_argument('--notebook', action='store_true',
                       help='Launch interactive notebook (requires Jupyter)')
    
    args = parser.parse_args()
    
    if args.notebook:
        create_research_notebook()
        return
    
    # Load configuration if provided
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = None
    
    # Initialize engine
    engine = QLCCE_Engine(config)
    
    # Run analysis
    if args.quick:
        print("Running quick analysis...")
        results = engine.run_full_analysis(n_steps=500)
    else:
        print("Running full analysis...")
        results = engine.run_full_analysis(n_steps=args.steps)
    
    # Visualize
    if args.full or not args.quick:
        engine.visualize_results(save_figures=True)
    
    # Export if requested
    if args.export:
        engine.export_results(args.export)
    elif args.full:
        engine.export_results()
    
    return results

# ==================== QUICKSTART ====================

if __name__ == "__main__":
    print("""
    ============================================================
    QUANTUM-LOGARITHMIC CHAOTIC CONSTRAINT ENGINE (QLCCE)
    ============================================================
    
    A research environment for studying:
    1. Quantum field theory simulations
    2. Benford's law compliance across scales
    3. Multi-base logarithmic transformations
    4. Chaotic systems with adaptive constraints
    
    Quick demo:
    """)
    
    # Run a quick demo
    engine = QLCCE_Engine()
    results = engine.run_full_analysis(n_steps=500)
    engine.visualize_results(save_figures=True)
