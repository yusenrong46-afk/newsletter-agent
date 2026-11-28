import {
  AlertTriangle,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronRight,
  Database,
  FileText,
  Loader2,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  Square,
  Wrench,
  X,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { ApprovalState, ChatMessage, Citation, ToolTrace, WorkbenchMode } from '../types'
import { MarkdownText } from './MarkdownText'

const STARTER_PROMPTS = [
  {
    title: 'Inventory overview',
    prompt: 'Give me an overview of my newsletter inventory and category totals.',
    tool: 'get_inventory_overview',
  },
  {
    title: 'Most unread clutter',
    prompt: 'Show me the publications with the most unread emails so I can unsubscribe.',
    tool: 'list_publications',
  },
  {
    title: 'Hybrid briefing search',
    prompt: 'Which newsletter mentioned river otters or solar sails?',
    tool: 'search_briefings',
  },
  {
    title: 'Approval-gated summary',
    prompt: 'Summarize Byte Garden from the last 7 days.',
    tool: 'summarize_newsletters',
  },
]

const TOOL_LABELS: Record<string, string> = {
  get_inventory_overview: 'Inventory Overview',
  list_publications: 'List Publications',
  search_briefings: 'Hybrid Briefing Search',
  summarize_newsletters: 'Summarize Newsletters (Gmail MCP)',
}

function ToolIcon({ name }: { name: string }) {
  if (name === 'search_briefings') return <Search size={14} />
  if (name === 'summarize_newsletters') return <Sparkles size={14} />
  if (name === 'get_inventory_overview') return <Database size={14} />
  return <Wrench size={14} />
}

function ToolTraceCard({ trace }: { trace: ToolTrace }) {
  const [open, setOpen] = useState(false)
  const label = TOOL_LABELS[trace.name] || trace.name

  return (
    <div className={`tool-card tool-status-${trace.status}`}>
      <button
        type="button"
        className="tool-card-header"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="tool-card-left">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <ToolIcon name={trace.name} />
          <span className="tool-card-name">{label}</span>
          <code className="tool-fn-name">{trace.name}</code>
        </span>
        <span className="tool-card-right">
          {trace.status === 'running' && (
            <span className="trace-badge running">
              <Loader2 size={12} className="spin" />
              Running
            </span>
          )}
          {trace.status === 'awaiting_approval' && (
            <span className="trace-badge warn">Awaiting approval</span>
          )}
          {trace.status === 'done' && (
            <span className="trace-badge ok">
              <Check size={12} />
              {trace.duration_ms !== undefined ? `${trace.duration_ms}ms` : 'Done'}
            </span>
          )}
          {trace.status === 'error' && <span className="trace-badge err">Error</span>}
        </span>
      </button>

      {open && (
        <div className="tool-card-body">
          <div className="trace-block">
            <div className="trace-label">Arguments</div>
            <pre className="trace-json">{JSON.stringify(trace.arguments, null, 2)}</pre>
          </div>
          {trace.output !== undefined && (
            <div className="trace-block">
              <div className="trace-label">Output</div>
              <pre className="trace-json">{JSON.stringify(trace.output, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ApprovalCard({
  approval,
  onDecide,
}: {
  approval: ApprovalState
  onDecide: (approvalId: string, approved: boolean) => void
}) {
  return (
    <div className={`approval-card approval-${approval.decision}`} role="region" aria-label="Tool approval request">
      <div className="approval-header">
        <div className="approval-title-row">
          {approval.decision === 'pending' ? (
            <AlertTriangle size={16} className="approval-icon-warn" />
          ) : approval.decision === 'approved' ? (
            <ShieldCheck size={16} className="approval-icon-ok" />
          ) : (
            <X size={16} className="approval-icon-muted" />
          )}
          <span className="approval-title">
            {approval.decision === 'pending'
              ? 'Approval required before reading Gmail'
              : approval.decision === 'approved'
                ? 'Approved — reading via local MCP'
                : 'Declined — nothing was read or sent'}
          </span>
        </div>
        <span className="approval-cap">
          Est. {approval.estimated_calls} call{approval.estimated_calls === 1 ? '' : 's'} ·{' '}
          {approval.estimated_tokens.toLocaleString()} tokens
        </span>
      </div>

      <p className="approval-summary">{approval.summary}</p>

      {approval.decision === 'pending' && (
        <div className="approval-actions">
          <button
            id={`approve-btn-${approval.approval_id}`}
            type="button"
            className="btn-primary"
            disabled={approval.submitting}
            onClick={() => onDecide(approval.approval_id, true)}
          >
            <Check size={14} />
            <span>{approval.submitting ? 'Submitting…' : 'Approve & Run'}</span>
          </button>
          <button
            id={`decline-btn-${approval.approval_id}`}
            type="button"
            className="btn-secondary"
            disabled={approval.submitting}
            onClick={() => onDecide(approval.approval_id, false)}
          >
            <X size={14} />
            <span>Decline</span>
          </button>
        </div>
      )}
    </div>
  )
}

interface ChatPanelProps {
  mode: WorkbenchMode
  modelName: string
  messages: ChatMessage[]
  busy: boolean
  onSend: (text: string) => void
  onStop: () => void
  onReset: () => void
  onApprove: (approvalId: string, approved: boolean) => void
  onCitationClick: (citation: Citation) => void
}

export function ChatPanel({
  mode,
  modelName,
  messages,
  busy,
  onSend,
  onStop,
  onReset,
  onApprove,
  onCitationClick,
}: ChatPanelProps) {
  const [draft, setDraft] = useState('')
  const endRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const submit = () => {
    const trimmed = draft.trim()
    if (!trimmed || busy) return
    setDraft('')
    onSend(trimmed)
  }

  return (
    <section className="workspace-panel chat-panel" aria-label="Agent Chat">
      <div className="panel-topbar">
        <div className="panel-topbar-meta">
          <span className="panel-title">Agent Chat</span>
          <span className="panel-subtitle">
            {mode === 'demo'
              ? 'Offline tool-calling agent · real local tools & in-memory MCP'
              : `Live agent · ${modelName} · read-only Gmail MCP`}
          </span>
        </div>
        {messages.length > 0 && (
          <button
            id="chat-reset-btn"
            type="button"
            className="btn-ghost-sm"
            onClick={onReset}
            disabled={busy}
          >
            <RotateCcw size={13} />
            <span>New chat</span>
          </button>
        )}
      </div>

      <div className="chat-scroll">
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <h2 className="welcome-heading">Ask your newsletter agent</h2>
            <p className="welcome-copy">
              Every count is computed in Python, every briefing bullet is fact-checked against its source email,
              and reading Gmail always pauses for your explicit approval.
            </p>
            <div className="starter-grid">
              {STARTER_PROMPTS.map((item) => (
                <button
                  key={item.tool}
                  type="button"
                  className="starter-card"
                  onClick={() => onSend(item.prompt)}
                  disabled={busy}
                >
                  <div className="starter-top">
                    <span className="starter-title">{item.title}</span>
                    <code className="starter-tool">{item.tool}</code>
                  </div>
                  <span className="starter-prompt">{item.prompt}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((msg) => (
              <article key={msg.id} className={`chat-turn turn-${msg.role}`}>
                <div className="turn-label">{msg.role === 'user' ? 'You' : 'Newsletter Agent'}</div>

                {msg.tools.length > 0 && (
                  <div className="tool-stack">
                    {msg.tools.map((trace) => (
                      <ToolTraceCard key={trace.call_id} trace={trace} />
                    ))}
                  </div>
                )}

                {msg.approvals.length > 0 && (
                  <div className="approval-stack">
                    {msg.approvals.map((apr) => (
                      <ApprovalCard key={apr.approval_id} approval={apr} onDecide={onApprove} />
                    ))}
                  </div>
                )}

                {msg.content && (
                  <div className={`turn-bubble bubble-${msg.role}`}>
                    {msg.role === 'assistant' ? <MarkdownText text={msg.content} /> : <p>{msg.content}</p>}
                  </div>
                )}

                {msg.streaming && !msg.content && msg.approvals.every((a) => a.decision !== 'pending') && (
                  <div className="streaming-indicator">
                    <Loader2 size={14} className="spin" />
                    <span>Thinking…</span>
                  </div>
                )}

                {msg.error && (
                  <div className="turn-error" role="alert">
                    <AlertTriangle size={14} />
                    <span>{msg.error}</span>
                  </div>
                )}

                {msg.citations.length > 0 && (
                  <div className="citation-strip" aria-label="Sources">
                    {msg.citations.map((cit, idx) => (
                      <button
                        key={`${cit.message_id}-${idx}`}
                        type="button"
                        className="citation-chip"
                        title={cit.snippet}
                        onClick={() => onCitationClick(cit)}
                      >
                        <FileText size={12} />
                        <span className="citation-pub">{cit.publication || 'Source'}</span>
                        {cit.received && <span className="citation-date">{cit.received.slice(0, 10)}</span>}
                      </button>
                    ))}
                  </div>
                )}

                {msg.role === 'assistant' && !msg.streaming && msg.usage && (
                  <div className="turn-footer">
                    {msg.tools.length > 0 && (
                      <span>
                        {msg.tools.length} tool call{msg.tools.length === 1 ? '' : 's'}
                      </span>
                    )}
                    <span>
                      {msg.usage.total_tokens > 0
                        ? `${msg.usage.total_tokens.toLocaleString()} tokens (${msg.usage.model_calls} model call${
                            msg.usage.model_calls === 1 ? '' : 's'
                          })`
                        : '0 tokens (offline demo)'}
                    </span>
                  </div>
                )}
              </article>
            ))}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <div className="composer-wrap">
        <div className="composer-box">
          <textarea
            id="chat-input"
            ref={inputRef}
            className="composer-textarea"
            rows={1}
            placeholder="Ask about unread newsletters, search past briefings, or summarize recent issues..."
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
          />
          {busy ? (
            <button
              id="chat-stop-btn"
              type="button"
              className="composer-btn stop"
              onClick={onStop}
              aria-label="Stop response"
            >
              <Square size={14} />
            </button>
          ) : (
            <button
              id="chat-send-btn"
              type="button"
              className="composer-btn send"
              disabled={!draft.trim()}
              onClick={submit}
              aria-label="Send message"
            >
              <ArrowUp size={16} />
            </button>
          )}
        </div>
        <div className="composer-footnote">
          Localhost only (`127.0.0.1`) · Raw email bodies are never written to disk · Gmail access is read-only
        </div>
      </div>
    </section>
  )
}
