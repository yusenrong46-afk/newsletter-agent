import { ArrowUpRight, BookOpen, CheckCircle2, Mail, Search, Sparkles, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { BriefingData, InventoryReport, PublicationGroup } from '../types'

type SidebarTab = 'publications' | 'briefings' | 'cleanup'

const CATEGORY_LABEL: Record<string, string> = {
  editorial_newsletter: 'Newsletter',
  promotional_subscription: 'Promotion',
  product_update: 'Product update',
}

interface SidebarProps {
  report: InventoryReport | null
  briefing: BriefingData | null
  indexedDocuments: number
  highlightedPublication: string | null
  onAskPrompt: (prompt: string) => void
  onSearchPublication: (publication: string) => void
}

export function Sidebar({
  report,
  briefing,
  indexedDocuments,
  highlightedPublication,
  onAskPrompt,
  onSearchPublication,
}: SidebarProps) {
  const [tab, setTab] = useState<SidebarTab>('publications')
  const [filter, setFilter] = useState('')

  const publications = useMemo<PublicationGroup[]>(() => {
    if (!report) return []
    return [...(report.publications_observed || []), ...(report.publications_provisional || [])]
  }, [report])

  const filteredPublications = useMemo(() => {
    const q = filter.trim().toLowerCase()
    if (!q) return publications
    return publications.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        p.category.toLowerCase().includes(q) ||
        (p.senders || []).some((s) => s.toLowerCase().includes(q)),
    )
  }, [publications, filter])

  const suggestions = report?.cleanup_advice?.suggestions || []
  const counts = report?.counts
  const unreadPct =
    counts && counts.subscription_messages > 0
      ? Math.round((100 * counts.subscription_unread) / counts.subscription_messages)
      : 0

  return (
    <aside className="sidebar" aria-label="Inbox context">
      <div className="sidebar-stats">
        <div className="stat-cell">
          <span className="stat-value">{counts ? counts.publications_observed : '—'}</span>
          <span className="stat-label">Publications</span>
        </div>
        <div className="stat-cell">
          <span className="stat-value">{counts ? counts.subscription_messages : '—'}</span>
          <span className="stat-label">Messages</span>
        </div>
        <div className="stat-cell">
          <span className="stat-value">{counts ? `${unreadPct}%` : '—'}</span>
          <span className="stat-label">Unread</span>
        </div>
        <div className="stat-cell">
          <span className="stat-value">{indexedDocuments}</span>
          <span className="stat-label">Indexed</span>
        </div>
      </div>

      {report?.reconciliation?.ok && (
        <div className="reconcile-banner">
          <CheckCircle2 size={14} />
          <span>Counts verified in Python · {report.run.window_start.slice(0, 10)} → {report.run.window_end.slice(0, 10)}</span>
        </div>
      )}

      <div className="sidebar-tabs" role="tablist" aria-label="Context views">
        <button
          id="sidebar-tab-publications"
          type="button"
          role="tab"
          aria-selected={tab === 'publications'}
          className={`sidebar-tab ${tab === 'publications' ? 'active' : ''}`}
          onClick={() => setTab('publications')}
        >
          <Mail size={14} />
          <span>Publications</span>
        </button>
        <button
          id="sidebar-tab-briefings"
          type="button"
          role="tab"
          aria-selected={tab === 'briefings'}
          className={`sidebar-tab ${tab === 'briefings' ? 'active' : ''}`}
          onClick={() => setTab('briefings')}
        >
          <BookOpen size={14} />
          <span>Briefing</span>
        </button>
        <button
          id="sidebar-tab-cleanup"
          type="button"
          role="tab"
          aria-selected={tab === 'cleanup'}
          className={`sidebar-tab ${tab === 'cleanup' ? 'active' : ''}`}
          onClick={() => setTab('cleanup')}
        >
          <Trash2 size={14} />
          <span>Cleanup</span>
          {suggestions.length > 0 && <span className="tab-count">{suggestions.length}</span>}
        </button>
      </div>

      <div className="sidebar-scroll">
        {tab === 'publications' && (
          <div className="sidebar-section">
            <div className="sidebar-search-wrap">
              <Search size={14} className="sidebar-search-icon" />
              <input
                id="sidebar-pub-filter"
                type="search"
                className="sidebar-search-input"
                placeholder="Filter publications..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                aria-label="Filter publications"
              />
            </div>

            {filteredPublications.length === 0 ? (
              <p className="empty-hint">
                {report ? 'No publications match that filter.' : 'Run `newsletter-agent inventory` or `--demo`.'}
              </p>
            ) : (
              <ul className="pub-list">
                {filteredPublications.map((pub) => {
                  const pct = pub.received_count > 0 ? Math.round((100 * pub.unread_count) / pub.received_count) : 0
                  const isHighlighted =
                    highlightedPublication &&
                    pub.name.toLowerCase() === highlightedPublication.toLowerCase()
                  return (
                    <li
                      key={pub.group_key}
                      className={`pub-item ${isHighlighted ? 'highlighted' : ''}`}
                    >
                      <div className="pub-header">
                        <span className="pub-name">{pub.name}</span>
                        <span className={`pill pill-${pub.category}`}>
                          {CATEGORY_LABEL[pub.category] || pub.category}
                        </span>
                      </div>
                      <div className="pub-meta">
                        <span>
                          {pub.received_count} received · {pub.unread_count} unread ({pct}%)
                        </span>
                        {pub.trend && <span className="pub-trend">{pub.trend}</span>}
                      </div>
                      <div className="unread-bar-track" aria-hidden="true">
                        <div
                          className={`unread-bar-fill ${pct >= 80 ? 'warn' : ''}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <div className="pub-actions">
                        <button
                          type="button"
                          className="ghost-action"
                          onClick={() => onSearchPublication(pub.name)}
                        >
                          <Search size={12} />
                          <span>Search</span>
                        </button>
                        <button
                          type="button"
                          className="ghost-action"
                          onClick={() => onAskPrompt(`Summarize ${pub.name} from the last 7 days`)}
                        >
                          <Sparkles size={12} />
                          <span>Summarize</span>
                        </button>
                      </div>
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        )}

        {tab === 'briefings' && (
          <div className="sidebar-section">
            {!briefing ? (
              <p className="empty-hint">No briefing saved yet. Run `newsletter-agent brief` or `--demo`.</p>
            ) : (
              <>
                <p className="briefing-window">
                  {briefing.counts.summarized} issues · {briefing.counts.publications} newsletters
                </p>
                {briefing.top_stories.length > 0 && (
                  <div className="briefing-group">
                    <h3 className="sidebar-subheading">Top stories</h3>
                    {briefing.top_stories.map((story, idx) => (
                      <div key={`story-${idx}`} className="story-card">
                        <div className="story-headline">{story.headline}</div>
                        <div className="story-meta">
                          {story.publication_count > 1
                            ? `Covered by ${story.publication_count} newsletters: `
                            : ''}
                          {story.publications.join(', ')}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                <div className="briefing-group">
                  <h3 className="sidebar-subheading">Fact-checked bullets</h3>
                  {briefing.sections.map((section) => (
                    <div key={section.publication} className="briefing-pub-block">
                      <div className="briefing-pub-title">{section.publication}</div>
                      {section.issues.map((issue) => (
                        <div key={issue.message_id} className="briefing-issue">
                          <ul className="briefing-bullets">
                            {issue.bullets.map((b, i) => (
                              <li key={`${issue.message_id}-${i}`}>{b}</li>
                            ))}
                          </ul>
                          <div className="briefing-source">{issue.subject}</div>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {tab === 'cleanup' && (
          <div className="sidebar-section">
            <p className="cleanup-note">
              Read-only suggestions ranked by unread volume. Newsletter Agent never unsubscribes or edits Gmail.
            </p>
            {suggestions.length === 0 ? (
              <p className="empty-hint">No cleanup suggestions for this window.</p>
            ) : (
              <ul className="cleanup-list">
                {suggestions.map((item) => (
                  <li key={item.name} className="cleanup-item">
                    <div className="cleanup-top">
                      <strong>{item.name}</strong>
                      <span
                        className={`pill ${
                          item.action.toLowerCase().includes('unsubscribe') ? 'pill-warn' : 'pill-info'
                        }`}
                      >
                        {item.action}
                      </span>
                    </div>
                    <p className="cleanup-reason">{item.reason}</p>
                    <button
                      type="button"
                      className="ghost-action"
                      onClick={() => onAskPrompt(`Did ${item.name} send anything worth reading recently?`)}
                    >
                      <span>Ask agent</span>
                      <ArrowUpRight size={12} />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </aside>
  )
}
