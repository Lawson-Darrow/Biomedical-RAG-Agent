import { useEffect, useState } from 'react'

const API = 'http://localhost:8000'

const DECISION_COLOR = { yes: '#16794b', no: '#b42318', maybe: '#b25e09' }

export default function App() {
  const [models, setModels] = useState([])
  const [model, setModel] = useState('')
  const [question, setQuestion] = useState('Does physical activity reduce the risk of type 2 diabetes?')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API}/models`)
      .then((r) => r.json())
      .then((d) => {
        setModels(d.models)
        setModel(d.default)
      })
      .catch(() => setError('Cannot reach the API. Is the backend running on :8000?'))
  }, [])

  async function ask() {
    if (!question.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const r = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, model }),
      })
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
      setResult(await r.json())
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="wrap">
      <header>
        <h1>Biomedical RAG Agent</h1>
        <p className="sub">
          Grounded answers from open-access literature, with citations. Evidence tool for
          clinicians/researchers — <strong>not medical advice</strong>.
        </p>
      </header>

      <div className="controls">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          placeholder="Ask a biomedical question..."
        />
        <div className="row">
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <button onClick={ask} disabled={loading}>
            {loading ? 'Thinking…' : 'Ask'}
          </button>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result">
          {result.abstained ? (
            <div className="badge abstain">Insufficient evidence — abstained</div>
          ) : (
            <div className="badge" style={{ background: DECISION_COLOR[result.decision] || '#444' }}>
              {result.decision?.toUpperCase()}
            </div>
          )}

          <p className="answer">{result.answer}</p>

          {result.citations.length > 0 && (
            <div className="citations">
              <h3>Citations</h3>
              {result.citations.map((c) => (
                <div className="cite" key={c.n}>
                  <span className="cnum">[{c.n}]</span>
                  <div>
                    <a
                      href={`https://pubmed.ncbi.nlm.nih.gov/${c.doc_id}/`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      PMID {c.doc_id}
                    </a>
                    <p>{c.passage}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
          <div className="meta">model: {result.model}</div>
        </div>
      )}
    </div>
  )
}
