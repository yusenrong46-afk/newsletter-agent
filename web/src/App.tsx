import { Bot, Loader2, Moon, PanelLeft, Search, Shield, Sun } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getBriefings, getInventory, getStatus, resetSession, streamChat, submitApproval } from './api'
import { ChatPanel } from './components/ChatPanel'
import { SearchPanel } from './components/SearchPanel'
import { Sidebar } from './components/Sidebar'
import type {
  AgentEvent,
  BriefingData,
  ChatMessage,
  Citation,
  InventoryReport,
  SearchHit,
  StatusResponse,
} from './types'

type ActiveView = 'chat' | 'search'

function makeSessionId(): string {
  return `sess_${Math.random().toString(36).slice(2, 10)}`
}

export function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [report, setReport] = useState<InventoryReport | null>(null)
  const [briefing, setBriefing] = useState<BriefingData | null>(null)
  const [loadingInit, setLoadingInit] = useState(true)
  const [initError, setInitError] = useState<string | null>(null)

  const [activeView, setActiveView] = useState<ActiveView>('chat')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = window.localStorage.getItem('na-theme')
    if (saved === 'light' || saved === 'dark') return saved
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  const [sessionId, setSessionId] = useState<string>(makeSessionId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const [highlightedPub, setHighlightedPub] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('central bank monetary policy cpi')
  const [searchPub, setSearchPub] = useState('')

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    window.localStorage.setItem('na-theme', theme)
  }, [theme])

  useEffect(() => {
    let cancelled = false
    Promise.all([getStatus(), getInventory(), getBriefings()])
      .then(([st, inv, br]) => {
        if (cancelled) return
        setStatus(st)
        setReport(inv.report)
        setBriefing(br.latest)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setInitError(err instanceof Error ? err.message : 'Failed to connect to local server.')
      })
      .finally(() => {
        if (!cancelled) setLoadingInit(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const publicationNames = useMemo(() => {
    if (!report) return []
    const all = [...(report.publications_observed || []), ...(report.publications_provisional || [])]
    return all.map((p) => p.name)
  }, [report])

  const updateAssistantMessage = useCallback(
    (assistantId: string, updater: (msg: ChatMessage) => ChatMessage) => {
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? updater(m) : m)))
    },
    [],
  )

  const handleSend = useCallback(
    async (text: string) => {
      if (busy) return
      setActiveView('chat')
      const userId = `u_${Date.now()}`
      const assistantId = `a_${Date.now() + 1}`

      setMessages((prev) => [
        ...prev,
        { id: userId, role: 'user', content: text, tools: [], approvals: [], citations: [] },
        {
          id: assistantId,
          role: 'assistant',
          content: '',
          tools: [],
          approvals: [],
          citations: [],
          streaming: true,
        },
      ])

      const controller = new AbortController()
      abortRef.current = controller
      setBusy(true)

      const onEvent = (event: AgentEvent) => {
        if (event.type === 'session') {
          setSessionId(event.session_id)
          return
        }
        updateAssistantMessage(assistantId, (msg) => {
          switch (event.type) {
            case 'tool_call':
              return {
                ...msg,
                tools: [
                  ...msg.tools,
                  {
                    call_id: event.call_id,
                    name: event.name,
                    arguments: event.arguments,
                    status: 'running',
                  },
                ],
              }
            case 'approval_required':
              return {
                ...msg,
                tools: msg.tools.map((t) =>
                  t.call_id === event.call_id ? { ...t, status: 'awaiting_approval' } : t,
                ),
                approvals: [
                  ...msg.approvals,
                  {
                    approval_id: event.approval_id,
                    call_id: event.call_id,
                    tool_name: event.tool_name,
                    summary: event.summary,
                    issue_count: event.issue_count,
                    estimated_calls: event.estimated_calls,
                    estimated_tokens: event.estimated_tokens,
                    decision: 'pending',
                  },
                ],
              }
            case 'approval_resolved':
              return {
                ...msg,
                tools: msg.tools.map((t) =>
                  t.status === 'awaiting_approval' ? { ...t, status: 'running' } : t,
                ),
                approvals: msg.approvals.map((a) =>
                  a.approval_id === event.approval_id
                    ? { ...a, decision: event.approved ? 'approved' : 'declined', submitting: false }
                    : a,
                ),
              }
            case 'tool_result':
              return {
                ...msg,
                tools: msg.tools.map((t) =>
                  t.call_id === event.call_id
                    ? {
                        ...t,
                        status: event.output && 'error' in event.output ? 'error' : 'done',
                        duration_ms: event.duration_ms,
                        output: event.output,
                      }
                    : t,
                ),
              }
            case 'token':
              return { ...msg, content: msg.content + event.delta }
            case 'done':
              return {
                ...msg,
                content: msg.content || event.answer,
                citations: event.citations || [],
                usage: event.usage,
                streaming: false,
              }
            case 'error':
              return { ...msg, error: event.message, streaming: false }
            default:
              return msg
          }
        })
      }

      try {
        await streamChat(text, sessionId, onEvent, controller.signal)
      } catch (err: unknown) {
        if ((err as { name?: string })?.name !== 'AbortError') {
          const message = err instanceof Error ? err.message : 'Connection interrupted.'
          updateAssistantMessage(assistantId, (msg) => ({ ...msg, error: message, streaming: false }))
        } else {
          updateAssistantMessage(assistantId, (msg) => ({ ...msg, streaming: false }))
        }
      } finally {
        updateAssistantMessage(assistantId, (msg) => ({ ...msg, streaming: false }))
        setBusy(false)
        abortRef.current = null
      }
    },
    [busy, sessionId, updateAssistantMessage],
  )

  const handleStop = () => {
    abortRef.current?.abort()
  }

  const handleReset = async () => {
    if (busy) return
    await resetSession(sessionId).catch(() => {})
    setMessages([])
    setSessionId(makeSessionId())
  }

  const handleApprove = async (approvalId: string, approved: boolean) => {
    setMessages((prev) =>
      prev.map((m) => ({
        ...m,
        approvals: m.approvals.map((a) => (a.approval_id === approvalId ? { ...a, submitting: true } : a)),
      })),
    )
    try {
      await submitApproval(approvalId, approved)
    } catch (err: unknown) {
      const detail = err instanceof Error ? err.message : 'Approval request failed'
      setMessages((prev) =>
        prev.map((m) =>
          m.approvals.some((a) => a.approval_id === approvalId)
            ? {
                ...m,
                error: detail,
                approvals: m.approvals.map((a) =>
                  a.approval_id === approvalId ? { ...a, submitting: false } : a,
                ),
              }
            : m,
        ),
      )
    }
  }

  const handleCitationClick = (citation: Citation) => {
    if (citation.publication) {
      setHighlightedPub(citation.publication)
      setSidebarOpen(true)
    }
  }

  const handleSearchPublication = (pub: string) => {
    setSearchPub(pub)
    setSearchQuery(pub)
    setActiveView('search')
  }

  const handleAskAboutHit = (hit: SearchHit) => {
    const prompt = hit.publication
      ? `Tell me more about what ${hit.publication} said regarding "${hit.subject || hit.text.slice(0, 60)}"`
      : `Search my briefings for ${hit.subject || hit.text.slice(0, 60)}`
    void handleSend(prompt)
  }

  if (loadingInit) {
    return (
      <div className="boot-screen">
        <Loader2 size={22} className="spin" />
        <p>Starting Newsletter Agent Workbench…</p>
      </div>
    )
  }

  if (initError) {
    return (
      <div className="boot-screen">
        <h1>Could not reach local workbench server</h1>
        <p>{initError}</p>
        <code>uv run newsletter-agent ui --demo</code>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-left">
          <button
            id="toggle-sidebar-btn"
            type="button"
            className="icon-btn"
            onClick={() => setSidebarOpen((v) => !v)}
            aria-label={sidebarOpen ? 'Hide inbox sidebar' : 'Show inbox sidebar'}
          >
            <PanelLeft size={17} />
          </button>
          <div className="brand-group">
            <h1 className="brand-title">Newsletter Agent</h1>
            <span className={`mode-badge mode-${status?.mode || 'demo'}`}>
              <Shield size={11} />
              {status?.mode === 'demo' ? 'DEMO MODE' : 'LIVE LOCAL'}
            </span>
          </div>
        </div>

        <nav className="view-switcher" aria-label="Workspace views">
          <button
            id="nav-chat-btn"
            type="button"
            className={`view-btn ${activeView === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveView('chat')}
          >
            <Bot size={15} />
            <span>Agent Chat</span>
          </button>
          <button
            id="nav-search-btn"
            type="button"
            className={`view-btn ${activeView === 'search' ? 'active' : ''}`}
            onClick={() => setActiveView('search')}
          >
            <Search size={15} />
            <span>Hybrid Search</span>
          </button>
        </nav>

        <div className="header-right">
          <span className="model-chip" title="Active agent backend">
            {status?.model || 'offline-demo-agent'}
          </span>
          <button
            id="theme-toggle-btn"
            type="button"
            className="icon-btn"
            onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
            aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
          >
            {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
          </button>
        </div>
      </header>

      <main className={`workbench-grid ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
        {sidebarOpen && (
          <Sidebar
            report={report}
            briefing={briefing}
            indexedDocuments={status?.indexed_documents || 0}
            highlightedPublication={highlightedPub}
            onAskPrompt={(prompt) => void handleSend(prompt)}
            onSearchPublication={handleSearchPublication}
          />
        )}

        <div className="workspace-main">
          {activeView === 'chat' ? (
            <ChatPanel
              mode={status?.mode || 'demo'}
              modelName={status?.model || 'offline-demo-agent'}
              messages={messages}
              busy={busy}
              onSend={(text) => void handleSend(text)}
              onStop={handleStop}
              onReset={() => void handleReset()}
              onApprove={(id, ok) => void handleApprove(id, ok)}
              onCitationClick={handleCitationClick}
            />
          ) : (
            <SearchPanel
              publications={publicationNames}
              query={searchQuery}
              publication={searchPub}
              onQueryChange={setSearchQuery}
              onPublicationChange={setSearchPub}
              onAskAboutHit={handleAskAboutHit}
            />
          )}
        </div>
      </main>
    </div>
  )
}

export default App
