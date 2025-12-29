"""
Quantum Field Sampler Module

Simulates quantum field fluctuations using lattice methods
with built-in Benford's law compliance checking.
"""

import numpy as np
from scipy import stats
import sympy as sp
from sympy.physics.quantum import Operator, Commutator


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
        lattice = np.random.randn(N, N).astype(np.float32)
        
        # Action: S = ∫ d²x [½(∂φ)² + ½m²φ² + (g/4!)φ⁴]
        beta = 1.0 / temperature
        
        for step in range(steps):
            # Update using checkerboard pattern
            for parity in range(2):
                for i in range(N):
                    for j in range(N):
                        if (i + j) % 2 == parity:
                            # Current action contribution
                            phi = lattice[i, j]
                            phi_nn = (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] + 
                                     lattice[i, (j+1)%N] + lattice[i, (j-1)%N]) / 4.0
                            
                            # Local action: ½(φ - φ_nn)² + ½m²φ² + (g/24)φ⁴
                            # Note: 24 = 4! is the factorial from φ⁴ theory quartic term
                            S_old = 0.5 * (phi - phi_nn)**2 + 0.5 * self.m**2 * phi**2 + (self.g / 24.0) * phi**4
                            
                            # Propose change
                            phi_new = phi + np.random.normal(0, 0.1)
                            S_new = 0.5 * (phi_new - phi_nn)**2 + 0.5 * self.m**2 * phi_new**2 + (self.g / 24.0) * phi_new**4
                            
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
            
            # Calculate deviation (KL divergence) with epsilon for numerical stability
            kl_div = np.sum(observed * np.log((observed + 1e-10) / (benford + 1e-10)))
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
