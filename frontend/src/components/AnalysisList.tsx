interface AnalysisListProps {
  analyses: { id: string; status: string }[]
  currentId: string | null
  onSelect: (id: string) => void
}

function AnalysisList({ analyses, currentId, onSelect }: AnalysisListProps) {
  if (analyses.length === 0) {
    return null
  }

  return (
    <div className="card">
      <h3>Previous Analyses</h3>
      <div className="mt-2" style={{ maxHeight: '200px', overflowY: 'auto' }}>
        {analyses.map((analysis) => (
          <div
            key={analysis.id}
            onClick={() => analysis.status === 'completed' && onSelect(analysis.id)}
            style={{
              padding: '0.5rem',
              margin: '0.25rem 0',
              backgroundColor: analysis.id === currentId ? '#4c3f91' : '#2d2d44',
              borderRadius: '4px',
              cursor: analysis.status === 'completed' ? 'pointer' : 'not-allowed',
              opacity: analysis.status === 'completed' ? 1 : 0.6,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span style={{ fontFamily: 'monospace', fontSize: '0.85em' }}>
              {analysis.id.slice(0, 8)}...
            </span>
            <span className={`status-${analysis.status}`} style={{ fontSize: '0.85em' }}>
              {analysis.status}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default AnalysisList
