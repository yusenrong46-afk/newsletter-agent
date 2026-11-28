import { ArrowUpRight, Loader2, Search, SlidersHorizontal } from 'lucide-react'
import { useEffect, useState } from 'react'
import { searchIndex } from '../api'
import type { SearchHit, SearchMode } from '../types'

const SAMPLE_QUERIES = [
  { label: 'Semantic (0 exact overlap)', query: 'central bank monetary policy cpi' },
  { label: 'Wildlife conservation', query: 'wildlife habitat restoration ecology' },
  { label: 'Space propulsion', query: 'solar sails spacecraft propulsion' },
  { label: 'Exact product version', query: 'Pixel Forge 4.0 release notes' },
]

const DOC_TYPE_LABEL: Record<string, string> = {
  briefing_bullet: 'Briefing bullet',
  top_story: 'Top story',
  issue_subject: 'Issue subject',
}

interface SearchPanelProps {
  publications: string[]
  query: string
  publication: string
  onQueryChange: (query: string) => void
  onPublicationChange: (publication: string) => void
  onAskAboutHit: (hit: SearchHit) => void
}

export function SearchPanel({
  publications,
  query,
  publication,
  onQueryChange,
  onPublicationChange,
  onAskAboutHit,
}: SearchPanelProps) {
  const [mode, setMode] = useState<SearchMode>('hybrid')
  const [days, setDays] = useState<string>('')
  const [searchState, setSearchState] = useState<{
    queryKey: string
    results: SearchHit[]
    error: string | null
  }>({ queryKey: '', results: [], error: null })

  const trimmed = query.trim()
  const currentKey = trimmed ? `${trimmed}|${mode}|${publication}|${days}` : ''
  const loading = Boolean(currentKey) && searchState.queryKey !== currentKey
  const results = currentKey ? searchState.results : []
  const error = currentKey ? searchState.error : null
  const hasSearched = Boolean(currentKey) && searchState.queryKey === currentKey

  useEffect(() => {
    if (!currentKey) return
    let cancelled = false
    const timer = window.setTimeout(() => {
      searchIndex({
        q: trimmed,
        mode,
        publication: publication || undefined,
        days: days ? Number(days) : null,
        limit: 12,
      })
        .then((res) => {
          if (!cancelled) {
            setSearchState({ queryKey: currentKey, results: res.results, error: null })
          }
        })
        .catch((err: unknown) => {
          if (!cancelled) {
            setSearchState({
              queryKey: currentKey,
              results: [],
              error: err instanceof Error ? err.message : 'Search failed',
            })
          }
        })
    }, 140)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [currentKey, trimmed, mode, publication, days])

  return (
    <section className="workspace-panel search-panel" aria-label="Hybrid Semantic Search">
      <div className="panel-topbar">
        <div className="panel-topbar-meta">
          <span className="panel-title">Hybrid Semantic Search</span>
          <span className="panel-subtitle">
            SQLite FTS5 (BM25) + dense vector cosine fused via Reciprocal Rank Fusion (k=60)
          </span>
        </div>
      </div>

      <div className="search-controls-area">
        <div className="search-input-bar">
          <Search size={16} className="search-bar-icon" />
          <input
            id="hybrid-search-input"
            type="search"
            className="search-bar-input"
            placeholder="Search fact-checked briefing bullets, top stories, and newsletter subjects..."
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            aria-label="Search indexed newsletters"
          />
          {loading && <Loader2 size={15} className="spin search-bar-spinner" />}
        </div>

        <div className="search-filter-row">
          <div className="mode-toggle" role="group" aria-label="Ranking mode">
            {(['hybrid', 'lexical', 'vector'] as SearchMode[]).map((m) => (
              <button
                key={m}
                id={`search-mode-${m}`}
                type="button"
                className={`mode-btn ${mode === m ? 'active' : ''}`}
                onClick={() => setMode(m)}
              >
                {m === 'hybrid' ? 'Hybrid RRF' : m === 'lexical' ? 'BM25 Lexical' : 'Vector Cosine'}
              </button>
            ))}
          </div>

          <div className="search-selects">
            <SlidersHorizontal size={14} className="select-icon" />
            <select
              id="search-pub-select"
              className="control-select"
              value={publication}
              onChange={(e) => onPublicationChange(e.target.value)}
              aria-label="Filter by publication"
            >
              <option value="">All publications</option>
              {publications.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <select
              id="search-days-select"
              className="control-select"
              value={days}
              onChange={(e) => setDays(e.target.value)}
              aria-label="Filter by recency"
            >
              <option value="">All time</option>
              <option value="7">Last 7 days</option>
              <option value="30">Last 30 days</option>
              <option value="90">Last 90 days</option>
            </select>
          </div>
        </div>

        <div className="sample-query-row">
          <span className="sample-label">Try comparing modes:</span>
          {SAMPLE_QUERIES.map((item) => (
            <button
              key={item.query}
              type="button"
              className={`sample-chip ${query === item.query ? 'active' : ''}`}
              onClick={() => onQueryChange(item.query)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="search-results-scroll">
        {error && (
          <div className="turn-error" role="alert">
            {error}
          </div>
        )}

        {!loading && hasSearched && results.length === 0 && (
          <div className="search-empty">
            <p className="search-empty-title">No matches in {mode.toUpperCase()} mode.</p>
            {mode === 'lexical' && (
              <p className="search-empty-hint">
                Pure BM25 requires exact keyword overlap. Switch to <strong>Hybrid RRF</strong> to retrieve
                semantically related bullets.
              </p>
            )}
          </div>
        )}

        {results.length > 0 && (
          <div className="result-list">
            {results.map((hit, idx) => (
              <article key={hit.doc_id} className="result-card">
                <div className="result-top">
                  <div className="result-meta-left">
                    <span className="result-rank">#{idx + 1}</span>
                    <span className="result-pub">{hit.publication || 'Unknown'}</span>
                    <span className={`pill pill-doc-${hit.doc_type}`}>
                      {DOC_TYPE_LABEL[hit.doc_type] || hit.doc_type}
                    </span>
                    {hit.received && <span className="result-date">{hit.received.slice(0, 10)}</span>}
                  </div>
                  <button
                    type="button"
                    className="ghost-action"
                    onClick={() => onAskAboutHit(hit)}
                  >
                    <span>Ask agent</span>
                    <ArrowUpRight size={12} />
                  </button>
                </div>

                <p className="result-text">{hit.text}</p>
                {hit.subject && !hit.text.includes(hit.subject) && (
                  <p className="result-subject">Subject: {hit.subject}</p>
                )}

                <div className="score-strip" aria-label="Ranking scores">
                  <span className="score-pill primary">RRF {hit.rrf_score.toFixed(4)}</span>
                  <span className="score-pill">
                    BM25 {hit.lexical_rank !== null ? `#${hit.lexical_rank}` : '—'}
                  </span>
                  <span className="score-pill">
                    Cosine {hit.vector_score !== null ? hit.vector_score.toFixed(3) : '—'}
                  </span>
                  <span className="score-pill muted">id: {hit.message_id}</span>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
