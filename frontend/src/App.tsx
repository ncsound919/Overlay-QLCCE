import { useState, useEffect, useCallback } from 'react'
import ConfigPanel from './components/ConfigPanel'
import ResultsPanel from './components/ResultsPanel'
import AnalysisList from './components/AnalysisList'

interface AnalysisConfig {
  lattice_size: number
  field_mass: number
  field_coupling: number
  chaos_system: string
  log_bases: string[]
  constraints: string[]
  n_steps: number
}

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

interface ConfigOptions {
  chaos_systems: string[]
  log_bases: string[]
  constraints: string[]
  lattice_size_range: { min: number; max: number; step: number }
  field_mass_range: { min: number; max: number; step: number }
  field_coupling_range: { min: number; max: number; step: number }
  n_steps_range: { min: number; max: number; step: number }
}

const defaultConfig: AnalysisConfig = {
  lattice_size: 64,
  field_mass: 0.1,
  field_coupling: 1.0,
  chaos_system: 'lorenz',
  log_bases: ['e', '10', '2', 'golden'],
  constraints: ['boundary', 'quantum'],
  n_steps: 500,
}

function App() {
  const [config, setConfig] = useState<AnalysisConfig>(defaultConfig)
  const [configOptions, setConfigOptions] = useState<ConfigOptions | null>(null)
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [analyses, setAnalyses] = useState<{ id: string; status: string }[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch config options on mount
  useEffect(() => {
    fetch('/api/config/options')
      .then((res) => res.json())
      .then(setConfigOptions)
      .catch((err) => console.error('Failed to fetch config options:', err))
  }, [])

  // Poll for analysis status
  useEffect(() => {
    if (!currentAnalysisId) return

    const pollStatus = async () => {
      try {
        const res = await fetch(`/api/analysis/status/${currentAnalysisId}`)
        const data = await res.json()

        if (data.status === 'completed') {
          setAnalysisResult(data)
          setIsLoading(false)
        } else if (data.status === 'failed') {
          setError(data.error || 'Analysis failed')
          setIsLoading(false)
        } else {
          // Continue polling
          setTimeout(pollStatus, 2000)
        }
      } catch (err) {
        console.error('Polling error:', err)
        setTimeout(pollStatus, 5000)
      }
    }

    pollStatus()
  }, [currentAnalysisId])

  const fetchAnalyses = useCallback(async () => {
    try {
      const res = await fetch('/api/analysis/list')
      const data = await res.json()
      setAnalyses(data)
    } catch (err) {
      console.error('Failed to fetch analyses:', err)
    }
  }, [])

  useEffect(() => {
    fetchAnalyses()
  }, [fetchAnalyses])

  const runAnalysis = async () => {
    setIsLoading(true)
    setError(null)
    setAnalysisResult(null)

    try {
      const res = await fetch('/api/analysis/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })

      if (!res.ok) {
        throw new Error('Failed to start analysis')
      }

      const data = await res.json()
      setCurrentAnalysisId(data.id)
      fetchAnalyses()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setIsLoading(false)
    }
  }

  const loadAnalysis = async (id: string) => {
    setIsLoading(true)
    setError(null)

    try {
      const res = await fetch(`/api/analysis/status/${id}`)
      const data = await res.json()

      if (data.status === 'completed') {
        setAnalysisResult(data)
        setCurrentAnalysisId(id)
      } else {
        setError(`Analysis status: ${data.status}`)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analysis')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container">
      <header className="header">
        <h1>QLCCE</h1>
        <p>
          Quantum-Logarithmic Chaotic Constraint Engine - A research environment
          for studying quantum field theory, Benford's law, and chaotic systems
        </p>
      </header>

      <div className="grid grid-3">
        <div>
          <ConfigPanel
            config={config}
            options={configOptions}
            onChange={setConfig}
            onRun={runAnalysis}
            isLoading={isLoading}
          />
        </div>

        <div style={{ gridColumn: 'span 2' }}>
          {error && (
            <div className="card" style={{ backgroundColor: '#3d1f1f' }}>
              <h3 style={{ color: '#d9534f' }}>Error</h3>
              <p>{error}</p>
            </div>
          )}

          {isLoading && !analysisResult && (
            <div className="card text-center">
              <h3>Running Analysis...</h3>
              <div className="loader"></div>
              <p>This may take a few moments depending on the configuration.</p>
            </div>
          )}

          {analysisResult && (
            <ResultsPanel result={analysisResult} />
          )}

          {!isLoading && !analysisResult && !error && (
            <div className="card text-center">
              <h3>Welcome to QLCCE</h3>
              <p className="mt-2">
                Configure your analysis parameters and click "Run Analysis" to begin.
              </p>
              <p className="mt-2" style={{ color: '#888' }}>
                The engine will simulate quantum field fluctuations, analyze
                Benford's law compliance, run chaotic systems, and visualize the results.
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="mt-3">
        <AnalysisList
          analyses={analyses}
          currentId={currentAnalysisId}
          onSelect={loadAnalysis}
        />
      </div>
    </div>
  )
}

export default App
