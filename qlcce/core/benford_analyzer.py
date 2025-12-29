"""
Benford Quantum Analyzer Module

Advanced Benford's law analysis with quantum field correlations.
"""

import numpy as np
from scipy import stats


# Define constant for numerical stability
EPSILON = 1e-10


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
                    # Guard against division by zero when d=0 and k=0
                    denominator = 10*k + d
                    if denominator > 0:
                        prob += np.log10(1 + 1/denominator)
                probs[d] = prob
            return probs / probs.sum()
    
    def _chi_square_test(self, observed_counts, total, digit_position):
        """Chi-square goodness-of-fit test"""
        expected_counts = self.expected * total if digit_position == 1 else self._expected_nth_digit(digit_position) * total
        mask = expected_counts > 0
        expected_counts = expected_counts[mask]
        observed_counts = observed_counts[mask]
        
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
        """
        Kullback-Leibler divergence D_KL(P||Q) = sum P(i) log(P(i)/Q(i))
        
        If expected probability is positive while observed is zero, the theoretical 
        KL divergence is infinite. We handle this by returning inf. For numerical
        stability, epsilon is added to both distributions consistently.
        """
        expected = self.expected if digit_position == 1 else self._expected_nth_digit(digit_position)
        kl = 0.0
        for d in range(len(observed)):
            # If expected has probability but observed doesn't, KL divergence is infinite
            if observed[d] <= 0 and expected[d] > EPSILON:
                return np.inf
            # Use epsilon consistently to decide whether to include this bin
            if observed[d] > EPSILON or expected[d] > EPSILON:
                kl += (observed[d] + EPSILON) * np.log((observed[d] + EPSILON) / (expected[d] + EPSILON))
        return kl
    
    def _js_divergence(self, observed, digit_position):
        """Jensen-Shannon divergence (symmetric)"""
        expected = self.expected if digit_position == 1 else self._expected_nth_digit(digit_position)
        m = 0.5 * (observed + expected)
        
        # Calculate KL(P||M) and KL(Q||M) correctly
        kl_obs_m = 0.0
        kl_exp_m = 0.0
        for d in range(len(observed)):
            if observed[d] > 0:
                kl_obs_m += observed[d] * np.log((observed[d] + EPSILON) / (m[d] + EPSILON))
            if expected[d] > 0:
                kl_exp_m += expected[d] * np.log((expected[d] + EPSILON) / (m[d] + EPSILON))
        
        js = 0.5 * kl_obs_m + 0.5 * kl_exp_m
        return js
    
    def analyze_multiscale(self, data, scales=None):
        """
        Analyze Benford compliance at multiple scales (logarithmic bins)
        """
        if scales is None:
            scales = [1, 10, 100, 1000]
        
        results = {}
        data = np.array(data)
        
        # Filter out non-positive values before binning
        data = data[data > 0]
        
        if len(data) == 0:
            return results
        
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
        Analyze correlation between Benford compliance and quantum field properties.

        Parameters
        ----------
        field_data : array-like or dict
            Quantum field configuration. If a dict is provided, it must contain
            a 'lattice' entry with the field values.
        position_correlation : bool, optional
            Reserved for future use. Currently ignored and kept only for
            backwards-compatible API; it has no effect on the analysis.
        """
        # NOTE: `position_correlation` is currently a placeholder argument and
        # does not affect the analysis. It is kept for backward compatibility.
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
        if len(region_compliance) > 1 and len(region_properties) > 0:
            compliance_array = np.array(region_compliance)
            properties_array = np.array([list(p.values()) for p in region_properties])
            
            # Ensure both arrays have the same length
            if len(compliance_array) == len(properties_array):
                # Correlation matrix: compliance with each property
                # Stack compliance as first column, then properties
                data_matrix = np.column_stack([compliance_array.reshape(-1, 1), properties_array])
                correlation_matrix = np.corrcoef(data_matrix.T)
                
                return {
                    'region_compliance': region_compliance,
                    'region_properties': region_properties,
                    'correlation_matrix': correlation_matrix,
                    'mean_compliance': np.mean(region_compliance),
                    'std_compliance': np.std(region_compliance)
                }
        
        return None
