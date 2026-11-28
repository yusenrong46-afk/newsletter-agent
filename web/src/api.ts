import type {
  AgentEvent,
  BriefingsResponse,
  InventoryResponse,
  SearchMode,
  SearchResponse,
  StatusResponse,
} from './types'

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const data = (await response.json()) as { error?: string }
      if (data.error) detail = data.error
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export function getStatus(): Promise<StatusResponse> {
  return fetchJson<StatusResponse>('/api/status')
}

export function getInventory(): Promise<InventoryResponse> {
  return fetchJson<InventoryResponse>('/api/inventory')
}

export function getBriefings(): Promise<BriefingsResponse> {
  return fetchJson<BriefingsResponse>('/api/briefings')
}

export function searchIndex(params: {
  q: string
  mode: SearchMode
  publication?: string
  days?: number | null
  limit?: number
}): Promise<SearchResponse> {
  const qs = new URLSearchParams()
  qs.set('q', params.q)
  qs.set('mode', params.mode)
  if (params.publication) qs.set('publication', params.publication)
  if (params.days) qs.set('days', String(params.days))
  if (params.limit) qs.set('limit', String(params.limit))
  return fetchJson<SearchResponse>(`/api/search?${qs.toString()}`)
}

export function submitApproval(approvalId: string, approved: boolean): Promise<{ ok: boolean }> {
  return fetchJson<{ ok: boolean }>('/api/chat/approve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approval_id: approvalId, approved }),
  })
}

export function resetSession(sessionId: string): Promise<{ ok: boolean }> {
  return fetchJson<{ ok: boolean }>('/api/chat/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
}

export async function streamChat(
  message: string,
  sessionId: string,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  })
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const data = (await response.json()) as { error?: string }
      if (data.error) detail = data.error
    } catch {
      // ignore
    }
    throw new Error(detail)
  }
  if (!response.body) {
    throw new Error('Streaming response body is unavailable.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const processBlock = (rawBlock: string) => {
    const lines = rawBlock.split('\n')
    let eventName = 'message'
    const dataLines: string[] = []
    for (const line of lines) {
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim())
      }
    }
    if (!dataLines.length) return
    try {
      const parsed = JSON.parse(dataLines.join('\n')) as Record<string, unknown>
      const typed = { type: eventName, ...parsed } as AgentEvent
      onEvent(typed)
    } catch {
      // Ignore malformed chunk
    }
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    buffer = buffer.replace(/\r\n/g, '\n')
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary).trim()
      buffer = buffer.slice(boundary + 2)
      if (block) processBlock(block)
      boundary = buffer.indexOf('\n\n')
    }
  }

  if (buffer.trim()) {
    processBlock(buffer.trim())
  }
}
