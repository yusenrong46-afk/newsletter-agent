export type WorkbenchMode = 'demo' | 'live'
export type SearchMode = 'hybrid' | 'lexical' | 'vector'

export interface StatusResponse {
  mode: WorkbenchMode
  model: string
  has_api_key: boolean
  has_inventory: boolean
  has_briefing: boolean
  indexed_documents: number
  embedding_models: Record<string, number>
  tools: string[]
}

export interface PublicationGroup {
  group_key: string
  name: string
  category: string
  received_count: number
  unread_count: number
  status: string
  confidence?: string
  trend?: string
  senders?: string[]
  weekly_counts?: number[]
}

export interface CleanupSuggestion {
  name: string
  action: string
  reason: string
  received_count?: number
  unread_count?: number
}

export interface InventoryReport {
  run: {
    window_start: string
    window_end: string
    status: string
    report_generated_at: string
  }
  counts: {
    publications_observed: number
    publications_provisional: number
    subscription_messages: number
    subscription_unread: number
    discovered?: number
    classified?: number
  }
  category_breakdown: Record<string, number>
  publications_observed: PublicationGroup[]
  publications_provisional: PublicationGroup[]
  cleanup_advice?: {
    suggestions?: CleanupSuggestion[]
    note?: string
  }
  reconciliation?: {
    ok: boolean
    problems?: string[]
  }
}

export interface InventoryResponse {
  available: boolean
  report: InventoryReport | null
}

export interface BriefingIssue {
  message_id: string
  publication: string
  subject: string
  received: string
  internal_date?: number
  bullets: string[]
}

export interface BriefingSection {
  publication: string
  issues: BriefingIssue[]
}

export interface TopStory {
  headline: string
  publications: string[]
  publication_count: number
  bullets: Array<{
    bullet_id?: string
    publication: string
    source_message_id?: string
    text: string
  }>
}

export interface BriefingData {
  generated_at: string
  window: string
  inventory_status: string
  sections: BriefingSection[]
  top_stories: TopStory[]
  counts: {
    issues: number
    summarized: number
    publications: number
    dropped_bullets: number
    unsupported_bullets: number
    unchecked_bullets: number
    top_stories: number
  }
  notes: string[]
}

export interface BriefingSummary {
  date: string
  filename: string
  window: string
  summarized: number
  publications: number
  top_stories: number
}

export interface BriefingsResponse {
  available: boolean
  latest: BriefingData | null
  history: BriefingSummary[]
}

export interface SearchHit {
  doc_id: string
  doc_type: string
  message_id: string
  publication: string
  subject: string
  text: string
  received: string
  internal_date: number
  briefing_date: string
  score: number
  rrf_score: number
  lexical_rank: number | null
  vector_rank: number | null
  vector_score: number | null
}

export interface SearchResponse {
  query: string
  mode: SearchMode
  publication: string | null
  days: number | null
  total_results: number
  results: SearchHit[]
}

export interface Citation {
  publication: string
  subject: string
  received: string
  message_id: string
  doc_type: string
  snippet: string
}

export interface ToolTrace {
  call_id: string
  name: string
  arguments: Record<string, unknown>
  status: 'running' | 'awaiting_approval' | 'done' | 'error'
  duration_ms?: number
  output?: Record<string, unknown>
}

export interface ApprovalState {
  approval_id: string
  call_id: string
  tool_name: string
  summary: string
  issue_count: number
  estimated_calls: number
  estimated_tokens: number
  decision: 'pending' | 'approved' | 'declined'
  submitting?: boolean
}

export interface UsageInfo {
  model_calls: number
  input_tokens: number
  output_tokens: number
  total_tokens: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  tools: ToolTrace[]
  approvals: ApprovalState[]
  citations: Citation[]
  usage?: UsageInfo
  streaming?: boolean
  error?: string
}

export type AgentEvent =
  | { type: 'session'; session_id: string; mode: WorkbenchMode }
  | { type: 'tool_call'; call_id: string; name: string; arguments: Record<string, unknown> }
  | {
      type: 'approval_required'
      approval_id: string
      call_id: string
      tool_name: string
      summary: string
      issue_count: number
      estimated_calls: number
      estimated_tokens: number
    }
  | { type: 'approval_resolved'; approval_id: string; approved: boolean }
  | { type: 'tool_result'; call_id: string; name: string; output: Record<string, unknown>; duration_ms: number }
  | { type: 'token'; delta: string }
  | { type: 'done'; answer: string; citations: Citation[]; usage: UsageInfo }
  | { type: 'error'; message: string }
