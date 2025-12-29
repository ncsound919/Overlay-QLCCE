interface ConfigPanelProps {
  config: {
    lattice_size: number
    field_mass: number
    field_coupling: number
    chaos_system: string
    log_bases: string[]
    constraints: string[]
    n_steps: number
  }
  options: {
    chaos_systems: string[]
    log_bases: string[]
    constraints: string[]
    lattice_size_range: { min: number; max: number; step: number }
    field_mass_range: { min: number; max: number; step: number }
    field_coupling_range: { min: number; max: number; step: number }
    n_steps_range: { min: number; max: number; step: number }
  } | null
  onChange: (config: ConfigPanelProps['config']) => void
  onRun: () => void
  isLoading: boolean
}

function ConfigPanel({ config, options, onChange, onRun, isLoading }: ConfigPanelProps) {
  const handleChange = (key: string, value: number | string | string[]) => {
    onChange({ ...config, [key]: value })
  }

  const toggleArrayItem = (key: 'log_bases' | 'constraints', item: string) => {
    const current = config[key]
    const newValue = current.includes(item)
      ? current.filter((i) => i !== item)
      : [...current, item]
    handleChange(key, newValue)
  }

  return (
    <div className="card">
      <h2>Configuration</h2>

      <div className="form-group">
        <label>Lattice Size</label>
        <input
          type="range"
          min={options?.lattice_size_range.min ?? 16}
          max={options?.lattice_size_range.max ?? 128}
          step={options?.lattice_size_range.step ?? 16}
          value={config.lattice_size}
          onChange={(e) => handleChange('lattice_size', parseInt(e.target.value))}
        />
        <span>{config.lattice_size}</span>
      </div>

      <div className="form-group">
        <label>Field Mass</label>
        <input
          type="range"
          min={options?.field_mass_range.min ?? 0.01}
          max={options?.field_mass_range.max ?? 2}
          step={options?.field_mass_range.step ?? 0.01}
          value={config.field_mass}
          onChange={(e) => handleChange('field_mass', parseFloat(e.target.value))}
        />
        <span>{config.field_mass.toFixed(2)}</span>
      </div>

      <div className="form-group">
        <label>Field Coupling</label>
        <input
          type="range"
          min={options?.field_coupling_range.min ?? 0.1}
          max={options?.field_coupling_range.max ?? 5}
          step={options?.field_coupling_range.step ?? 0.1}
          value={config.field_coupling}
          onChange={(e) => handleChange('field_coupling', parseFloat(e.target.value))}
        />
        <span>{config.field_coupling.toFixed(1)}</span>
      </div>

      <div className="form-group">
        <label>Simulation Steps</label>
        <input
          type="range"
          min={options?.n_steps_range.min ?? 100}
          max={options?.n_steps_range.max ?? 2000}
          step={options?.n_steps_range.step ?? 100}
          value={config.n_steps}
          onChange={(e) => handleChange('n_steps', parseInt(e.target.value))}
        />
        <span>{config.n_steps}</span>
      </div>

      <div className="form-group">
        <label>Chaotic System</label>
        <select
          value={config.chaos_system}
          onChange={(e) => handleChange('chaos_system', e.target.value)}
        >
          {(options?.chaos_systems ?? ['lorenz', 'rossler', 'logistic']).map((sys) => (
            <option key={sys} value={sys}>
              {sys.charAt(0).toUpperCase() + sys.slice(1)}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Logarithmic Bases</label>
        <div className="checkbox-group">
          {(options?.log_bases ?? ['e', '10', '2', 'golden']).map((base) => (
            <label
              key={base}
              className={`checkbox-item ${config.log_bases.includes(base) ? 'selected' : ''}`}
            >
              <input
                type="checkbox"
                checked={config.log_bases.includes(base)}
                onChange={() => toggleArrayItem('log_bases', base)}
              />
              {base}
            </label>
          ))}
        </div>
      </div>

      <div className="form-group">
        <label>Constraints</label>
        <div className="checkbox-group">
          {(options?.constraints ?? ['boundary', 'quantum']).map((constraint) => (
            <label
              key={constraint}
              className={`checkbox-item ${config.constraints.includes(constraint) ? 'selected' : ''}`}
            >
              <input
                type="checkbox"
                checked={config.constraints.includes(constraint)}
                onChange={() => toggleArrayItem('constraints', constraint)}
              />
              {constraint}
            </label>
          ))}
        </div>
      </div>

      <button
        onClick={onRun}
        disabled={isLoading}
        style={{ width: '100%', marginTop: '1rem' }}
      >
        {isLoading ? 'Running...' : 'Run Analysis'}
      </button>
    </div>
  )
}

export default ConfigPanel
