import React, { useState } from 'react'
import { analyze } from './api'

export default function App() {
  const [jd, setJd] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    if (!file || !jd.trim()) {
      setError('Please paste a job description and upload a resume PDF.')
      return
    }
    setLoading(true)
    try {
      const data = await analyze(jd, file)
      setResult(data)
    } catch (err: any) {
      setError(err?.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{maxWidth: 800, margin: '0 auto', padding: 24}}>
      <h1>Resume ↔ JD Match</h1>
      <p>Paste a job description and upload your resume (PDF). Get a match score, missing skills and suggested bullets.</p>

      <form onSubmit={onSubmit} style={{display:'grid', gap:12}}>
        <textarea
          placeholder="Paste job description..."
          rows={10}
          value={jd}
          onChange={e=>setJd(e.target.value)}
          style={{width:'100%', padding:12, border:'1px solid #ccc', borderRadius:8}}
        />
        <input type="file" accept="application/pdf" onChange={e=>setFile(e.target.files?.[0] ?? null)} />
        <button disabled={loading} style={{padding:'10px 16px', borderRadius:8}}>
          {loading ? 'Scoring...' : 'Analyze'}
        </button>
      </form>

      {error && <div style={{marginTop:12, color:'crimson'}}>{error}</div>}

      {result && (
        <div style={{marginTop:24}}>
          <h2>Results</h2>
          <div><b>Match Score:</b> {result.score}/100</div>

          <div style={{marginTop:12}}>
            <b>Top Gaps</b>
            <div style={{display:'flex', gap:8, flexWrap:'wrap', marginTop:8}}>
              {result.missing_skills?.map((s: string, i: number) => (
                <span key={i} style={{border:'1px solid #ddd', padding:'4px 8px', borderRadius:999}}>{s}</span>
              ))}
            </div>
          </div>

          <div style={{marginTop:12}}>
            <b>Suggested Bullets</b>
            <ul>
              {result.suggested_bullets?.map((b: string, i: number) => <li key={i}>{b}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
