interface AnalysisResult {
  id: string
  status: string
  field_properties?: {
    mean: number
    variance: number
    skewness: number
    kurtosis: number
    entropy: number
  }
  benford_analysis?: {
    observed_distribution: number[]
    expected_distribution: number[]
    compliance_score: number
    chi2_p_value: number
  }
  lyapunov_exponents?: number[]
  log_periodicity?: {
    periods: number[]
    significance: number
    base: string
  }
  visualization?: string
}

interface ResultsPanelProps {
  result: AnalysisResult
}

function ResultsPanel({ result }: ResultsPanelProps) {
  return (
    <div className="results-section">
      <div className="card">
        <div className="flex justify-between items-center mb-2">
          <h2>Analysis Results</h2>
          <span className={`status-${result.status}`}>
            {result.status.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-4 mb-2">
        {result.field_properties && (
          <>
            <div className="card text-center">
              <label>Field Mean</label>
              <div className="stat-value">
                {result.field_properties.mean.toFixed(4)}
              </div>
            </div>
            <div className="card text-center">
              <label>Field Variance</label>
              <div className="stat-value">
                {result.field_properties.variance.toFixed(4)}
              </div>
            </div>
            <div className="card text-center">
              <label>Field Entropy</label>
              <div className="stat-value">
                {result.field_properties.entropy.toFixed(2)} bits
              </div>
            </div>
            <div className="card text-center">
              <label>Kurtosis</label>
              <div className="stat-value">
                {result.field_properties.kurtosis.toFixed(4)}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Benford Analysis */}
      {result.benford_analysis && (
        <div className="card">
          <h3>Benford's Law Analysis</h3>
          <div className="grid grid-2 mt-2">
            <div className="text-center">
              <label>Compliance Score</label>
              <div className="stat-value">
                {(result.benford_analysis.compliance_score * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-center">
              <label>Chi-Square p-value</label>
              <div className="stat-value">
                {result.benford_analysis.chi2_p_value.toExponential(3)}
              </div>
            </div>
          </div>
          
          <div className="mt-3">
            <h4 style={{ marginBottom: '0.5rem' }}>First Digit Distribution</h4>
            <div className="benford-chart">
              {result.benford_analysis.observed_distribution.map((obs, i) => {
                const exp = result.benford_analysis!.expected_distribution[i]
                const maxVal = Math.max(
                  ...result.benford_analysis!.observed_distribution,
                  ...result.benford_analysis!.expected_distribution
                )
                return (
                  <div key={i} className="benford-bar-group">
                    <div style={{ display: 'flex', alignItems: 'flex-end', height: '150px' }}>
                      <div
                        className="benford-bar observed"
                        style={{ height: `${(obs / maxVal) * 150}px` }}
                        title={`Observed: ${(obs * 100).toFixed(1)}%`}
                      />
                      <div
                        className="benford-bar expected"
                        style={{ height: `${(exp / maxVal) * 150}px` }}
                        title={`Expected: ${(exp * 100).toFixed(1)}%`}
                      />
                    </div>
                    <span className="bar-label">{i + 1}</span>
                  </div>
                )
              })}
            </div>
            <div className="flex justify-between mt-1" style={{ fontSize: '0.8em', color: '#888' }}>
              <span>■ Observed</span>
              <span>■ Expected (Benford)</span>
            </div>
          </div>
        </div>
      )}

      {/* Lyapunov Exponents */}
      {result.lyapunov_exponents && result.lyapunov_exponents.length > 0 && (
        <div className="card">
          <h3>Lyapunov Exponents</h3>
          <p style={{ color: '#888', fontSize: '0.9em', marginBottom: '0.5rem' }}>
            Positive values indicate chaotic behavior
          </p>
          <div className="flex flex-wrap">
            {result.lyapunov_exponents.map((exp, i) => (
              <div key={i} className="card text-center" style={{ minWidth: '100px' }}>
                <label>λ{i + 1}</label>
                <div
                  className="stat-value"
                  style={{ color: exp > 0 ? '#5cb85c' : '#d9534f' }}
                >
                  {exp.toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Log Periodicity */}
      {result.log_periodicity && result.log_periodicity.periods.length > 0 && (
        <div className="card">
          <h3>Log-Periodic Patterns</h3>
          <p style={{ color: '#888', fontSize: '0.9em' }}>
            Base: {result.log_periodicity.base} | 
            Significance: {(result.log_periodicity.significance * 100).toFixed(1)}%
          </p>
          <div className="mt-2">
            <strong>Detected Periods:</strong>{' '}
            {result.log_periodicity.periods.map((p) => p.toFixed(2)).join(', ')}
          </div>
        </div>
      )}

      {/* Visualization */}
      {result.visualization && (
        <div className="card">
          <h3>Visualization</h3>
          <div className="visualization-container mt-2">
            <img
              src={`data:image/png;base64,${result.visualization}`}
              alt="QLCCE Analysis Results"
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default ResultsPanel
