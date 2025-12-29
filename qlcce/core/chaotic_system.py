"""
Chaotic Constraint System Module

Chaotic systems with adaptive constraints and boundary conditions.
"""

import numpy as np


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
            
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            # Apply constraints if any (after computing derivatives)
            result = [dx, dy, dz]
            if constraints:
                for constraint in constraints:
                    result = constraint(result, t)
            
            return result
        
        return equations
    
    def rossler_system(self, params=None, constraints=None):
        """
        Rossler (ASCII representation of Rössler) system with optional constraints.
        
        Note: the function name uses ASCII 'rossler' for compatibility with code 
        and tooling; it corresponds to the Rössler chaotic system.
        """
        if params is None:
            params = {'a': 0.2, 'b': 0.2, 'c': 5.7}
        
        a, b, c = params['a'], params['b'], params['c']
        
        def equations(state, t):
            x, y, z = state
            
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            
            # Apply constraints (after computing derivatives)
            result = [dx, dy, dz]
            if constraints:
                for constraint in constraints:
                    result = constraint(result, t)
            
            return result
        
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
        def constraint(state, t=None, *args):
            # Handle both ODE systems (state, t) and logistic map (state, prev_state, iteration)
            if isinstance(state, (list, np.ndarray)) and not np.isscalar(state):
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
                if isinstance(bounds, dict) and 'bounds' in bounds:
                    low, high = bounds['bounds']
                    if s < low:
                        s = low + (low - s)
                    elif s > high:
                        s = high - (s - high)
                elif isinstance(bounds, (list, tuple)) and len(bounds) > 0:
                    # Handle bounds provided as list of tuples for single value
                    low, high = bounds[0]
                    if s < low:
                        s = low + (low - s)
                    elif s > high:
                        s = high - (s - high)
                return s
        
        return constraint
    
    def _create_symmetry_constraint(self, axis='z'):
        """Create symmetry constraint (e.g., z → -z)"""
        def constraint(state, t=None):
            # Avoid in-place modification so that chained constraints
            # do not unexpectedly share mutated state.
            if isinstance(state, np.ndarray):
                new_state = state.copy()
            elif isinstance(state, list):
                new_state = list(state)
            else:
                # For scalar or unsupported types, leave state unchanged.
                return state

            if axis == 'x' and len(new_state) > 0:
                new_state[0] = -new_state[0]
            elif axis == 'y' and len(new_state) > 1:
                new_state[1] = -new_state[1]
            elif axis == 'z' and len(new_state) > 2:
                new_state[2] = -new_state[2]

            return new_state
        return constraint
    
    def _create_conservation_constraint(self, conserved_quantity='energy'):
        """
        Create a conservation-law constraint.

        This constraint is currently not implemented. It is reserved for future
        extensions where a concrete conservation law (e.g., energy, momentum)
        will be computed from the state and enforced over time.
        
        Raises
        ------
        NotImplementedError
            Always raised as this constraint type is not yet implemented.
        """
        raise NotImplementedError(
            "Conservation constraints are not yet implemented. "
            "This feature is reserved for future development."
        )
    
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
        lyapunov_exponents = np.zeros(m)
        
        # For simplicity, use a basic estimation
        # In practice, use more robust algorithm
        for i in range(m):
            # Calculate divergence of nearby trajectories
            divergences = []
            # Ensure indices j and j+1 are within bounds of the embedded array
            max_j = min(n - 10, embedded.shape[0] - 1)
            # If max_j is not greater than the lower bound, there are not enough
            # samples to compute meaningful divergences in this dimension.
            if max_j <= 10:
                continue
            for j in range(10, max_j):
                ref_point = embedded[j, i]
                # Find nearest neighbor
                distances = np.abs(embedded[:, i] - ref_point)
                distances[j-5:j+5] = np.inf  # Exclude immediate neighbors
                nn_idx = np.argmin(distances)
                
                # Guard against nn_idx+1 exceeding the array bounds
                if nn_idx >= embedded.shape[0] - 1:
                    continue
                
                divergence = np.abs(embedded[j+1, i] - embedded[nn_idx+1, i])
                initial_dist = np.abs(embedded[j, i] - embedded[nn_idx, i])
                
                if initial_dist > 0:
                    divergences.append(np.log(divergence / initial_dist))
            
            if divergences:
                lyapunov_exponents[i] = np.mean(divergences) / dt
        
        return lyapunov_exponents
    
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
