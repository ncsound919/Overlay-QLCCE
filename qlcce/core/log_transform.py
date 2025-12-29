"""
Multi-Base Logarithmic Transformer Module

Multi-base logarithmic transformations with chaotic constraint adaptation.
"""

import numpy as np


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
            raise ValueError(
                f"Mode {mode} not recognized. Available modes: 'direct', 'iterated', 'super-log', 'chaotic'"
            )
    
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
                    
                    # Calculate symmetric KL divergence with epsilon for numerical stability
                    kl_ij = np.sum(pi * np.log((pi + 1e-10) / (pj + 1e-10)))
                    kl_ji = np.sum(pj * np.log((pj + 1e-10) / (pi + 1e-10)))
                    
                    divergence[i, j] = (kl_ij + kl_ji) / 2
        
        return divergence
    
    def detect_log_periodicity(self, data, base='e', threshold=0.01):
        """
        Detect log-periodic patterns (signatures of discrete scale invariance)
        
        Parameters:
        -----------
        data : array-like
            Input data
        base : str or float
            Logarithmic base
        threshold : float
            Threshold for peak detection
        
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
        """
        Find peaks in 1D data above a given threshold.
        
        A peak is defined as a local maximum where the value at index i is
        greater than both neighbors (i-1 and i+1) and exceeds the threshold.
        """
        peaks = []
        for i in range(1, len(data)-1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold:
                peaks.append(i)
        return peaks
    
    def _calculate_significance(self, data, periods, n_surrogates=1000):
        """Calculate significance using phase randomization"""
        # Generate surrogate data by randomizing phases
        
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
