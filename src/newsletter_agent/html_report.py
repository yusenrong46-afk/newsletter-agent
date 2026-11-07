"""Static HTML pages for the inventory report and the briefing.

Each page is one self-contained file: no server, no CDN, no remote images.
Every piece of email text is untrusted, so all of it goes through html.escape,
and a Content-Security-Policy stops the page from loading anything remote.
"""

from html import escape

from .classify import ALL_CATEGORIES, SUBSCRIPTION_CATEGORIES

CONTENT_SECURITY_POLICY = "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:"

CATEGORY_LABELS = {
    "editorial_newsletter": "Newsletter",
    "promotional_subscription": "Promotion",
    "product_update": "Product update",
    "transactional": "Transactional",
    "personal_or_other": "Personal / other",
    "uncertain": "Uncertain",
}

STYLE = """
:root {
  --bg: #f7f7f5; --surface: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --border: #e3e2dd;
  --accent: #2f5bd3; --accent-soft: #e8eefc; --good: #1f7a4d; --good-soft: #e3f3ea;
  --warn: #9a5b00; --warn-soft: #fbefdc; --bar: #2f5bd3;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141413; --surface: #1d1d1b; --text: #ecebe6; --muted: #a3a29b; --border: #33322f;
    --accent: #8aa8ff; --accent-soft: #1f2a48; --good: #6fd3a0; --good-soft: #17342a;
    --warn: #f0b45c; --warn-soft: #3a2a12; --bar: #8aa8ff;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 12px; }
.muted { color: var(--muted); } .small { font-size: 13px; }
.demo { display: inline-block; background: var(--warn-soft); color: var(--warn); border-radius: 6px;
  padding: 2px 8px; font-size: 12px; font-weight: 600; letter-spacing: .04em; margin-bottom: 8px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 20px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.card .value { font-size: 26px; font-weight: 650; } .card .label { color: var(--muted); font-size: 13px; }
.badge { display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: 12px; font-weight: 600; }
.badge.good { background: var(--good-soft); color: var(--good); }
.badge.warn { background: var(--warn-soft); color: var(--warn); }
.badge.info { background: var(--accent-soft); color: var(--accent); }
.bars { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.bar-row { display: grid; grid-template-columns: 140px 1fr 56px; gap: 10px; align-items: center; margin: 6px 0; }
.bar-track { background: var(--bg); border-radius: 4px; height: 12px; overflow: hidden; }
.bar-fill { background: var(--bar); height: 100%; border-radius: 4px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.controls input, .controls select { font: inherit; padding: 6px 10px; border: 1px solid var(--border);
  border-radius: 8px; background: var(--surface); color: var(--text); }
.controls input { flex: 1 1 220px; }
.table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; min-width: 640px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { font-size: 13px; color: var(--muted); font-weight: 600; cursor: pointer; user-select: none; white-space: nowrap; }
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
details summary { cursor: pointer; } details .detail { color: var(--muted); font-size: 13px; margin-top: 6px; }
ul.notes li { margin: 4px 0; }
.section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin: 12px 0; }
.section h3 { margin: 0 0 8px; font-size: 16px; }
.source { color: var(--muted); font-size: 13px; }
"""

SCRIPT = """
(function () {
  const table = document.getElementById('publications');
  if (!table) return;
  const body = table.tBodies[0];
  const rows = Array.from(body.rows);
  const search = document.getElementById('search');
  const category = document.getElementById('category');
  const status = document.getElementById('status');
  function applyFilters() {
    const text = search.value.toLowerCase();
    rows.forEach(function (row) {
      const visible = row.dataset.search.includes(text)
        && (category.value === 'all' || row.dataset.category === category.value
            || (category.value === 'subscription' && row.dataset.subscription === 'yes'))
        && (status.value === 'all' || row.dataset.status === status.value
            || (status.value === 'mostly_unread' && Number(row.dataset.unread) * 2 > Number(row.dataset.received)));
      row.hidden = !visible;
    });
  }
  [search, category, status].forEach(function (el) { el.addEventListener('input', applyFilters); });
  table.querySelectorAll('th[data-sort]').forEach(function (header) {
    let ascending = header.dataset.sort === 'name';
    header.addEventListener('click', function () {
      const key = header.dataset.sort;
      rows.sort(function (a, b) {
        const x = a.dataset[key], y = b.dataset[key];
        const result = isNaN(x) ? x.localeCompare(y) : Number(x) - Number(y);
        return ascending ? result : -result;
      });
      ascending = !ascending;
      rows.forEach(function (row) { body.appendChild(row); });
    });
  });
})();
"""


def page(title: str, body: str, demo: bool = False) -> str:
    """Wrap content in the shared page shell."""
    banner = '<div class="demo">DEMO DATA</div>' if demo else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{CONTENT_SECURITY_POLICY}">
<meta name="referrer" content="no-referrer">
<title>{escape(title)}</title>
<style>{STYLE}</style>
</head>
<body>
<main>
{banner}
{body}
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""


def card(value, label: str) -> str:
    return f'<div class="card"><div class="value">{escape(str(value))}</div><div class="label">{escape(label)}</div></div>'


STATUS_BADGES = {"complete": "good", "preflight": "info", "awaiting_approval": "info"}


def status_badge(status: str) -> str:
    return f'<span class="badge {STATUS_BADGES.get(status, "warn")}">{escape(status.replace("_", " "))}</span>'


def category_bars(categories: dict, classified: int) -> str:
    rows = []
    for name in ALL_CATEGORIES:
        count = categories.get(name, 0)
        width = round(100 * count / classified, 1) if classified else 0
        rows.append(
            f'<div class="bar-row"><span>{escape(CATEGORY_LABELS[name])}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width}%"></div></div>'
            f'<span class="num">{count}</span></div>'
        )
    return f'<div class="bars">{"".join(rows)}<p class="muted small">Share of {classified} classified messages. Labels are model judgments; counts are exact.</p></div>'


def publication_row(group: dict) -> str:
    example = group["example"]
    other = ", ".join(f"{CATEGORY_LABELS.get(k, k)} {v}" for k, v in sorted(group["other_messages_same_identity"].items()))
    details = [
        f"Senders: {escape(', '.join(group['senders']))}",
        f"Grouped by: {escape(group['grouping_basis'])}",
        f"Example: “{escape(example['subject'])}” · {escape(example['received'])} · message {escape(example['message_id'])}",
    ]
    if group["provisional_reason"]:
        details.append(f"Provisional because: {escape(group['provisional_reason'])}")
    if other:
        details.append(f"Other mail from this identity (not counted): {escape(other)}")
    if len(group["name_variants"]) > 1:
        details.append(f"Name variants: {escape(', '.join(group['name_variants']))}")
    status_class = "good" if group["status"] == "observed" else "warn"
    search_text = " ".join([group["name"], *group["senders"]]).lower()
    return (
        f'<tr data-name="{escape(group["name"].lower())}" data-category="{escape(group["category"] or "")}" '
        f'data-subscription="{"yes" if group["category"] in SUBSCRIPTION_CATEGORIES else "no"}" '
        f'data-status="{escape(group["status"])}" data-received="{group["received_count"]}" '
        f'data-unread="{group["unread_count"]}" data-confidence="{escape(group["confidence"] or "")}" '
        f'data-search="{escape(search_text)}">'
        f'<td><details><summary>{escape(group["name"])}</summary><div class="detail">{"<br>".join(details)}</div></details></td>'
        f'<td>{escape(CATEGORY_LABELS.get(group["category"], group["category"] or ""))}</td>'
        f'<td class="num">{group["received_count"]}</td><td class="num">{group["unread_count"]}</td>'
        f'<td>{escape(group["confidence"] or "")}</td>'
        f'<td><span class="badge {status_class}">{escape(group["status"])}</span></td></tr>'
    )


def render_html_report(report: dict, demo: bool = False) -> str:
    run, counts = report["run"], report["counts"]
    groups = report["publications_observed"] + report["publications_provisional"]
    groups.sort(key=lambda group: (-group["received_count"], group["name"].lower()))
    usage = report["usage"]["total_for_this_window"]
    reconciliation = report["reconciliation"]
    check = ('<span class="badge good">All totals match</span>' if reconciliation["ok"]
             else f'<span class="badge warn">Problem: {escape("; ".join(reconciliation["problems"]))}</span>')
    category_options = "".join(
        f'<option value="{name}">{escape(CATEGORY_LABELS[name])}</option>' for name in SUBSCRIPTION_CATEGORIES
    )
    body = f"""
<h1>Newsletter inventory</h1>
<p class="muted">{escape(run['window_start'])} → {escape(run['window_end'])} ({run['days']} days) · {status_badge(run['status'])} · {check}</p>
<div class="cards">
  {card(counts['publications_observed'], 'observed publications')}
  {card(counts['publications_provisional'], 'provisional publications')}
  {card(counts['subscription_messages'], 'subscription messages')}
  {card(counts['subscription_unread'], 'of those unread')}
  {card(counts['in_scope'], 'messages scanned')}
</div>

<h2>Categories</h2>
{category_bars(report['category_breakdown'], counts['classified'])}

<h2>Publications</h2>
<div class="controls">
  <input id="search" type="search" placeholder="Search name or sender" aria-label="Search publications">
  <select id="category" aria-label="Filter by category"><option value="all">All categories</option>{category_options}</select>
  <select id="status" aria-label="Filter by status"><option value="all">Observed + provisional</option>
    <option value="observed">Observed only</option><option value="provisional">Provisional only</option>
    <option value="mostly_unread">Mostly unread</option></select>
</div>
<div class="table-wrap"><table id="publications">
<thead><tr><th data-sort="name">Publication</th><th data-sort="category">Category</th>
<th class="num" data-sort="received">Received</th><th class="num" data-sort="unread">Unread</th>
<th data-sort="confidence">Confidence</th><th data-sort="status">Status</th></tr></thead>
<tbody>{"".join(publication_row(group) for group in groups)}</tbody>
</table></div>
<p class="muted small">Click a column to sort and a name for details. Observed groups have a List-ID or repeated unsubscribe evidence; provisional groups have weaker evidence.</p>

<h2>Coverage</h2>
<div class="cards">
  {card(counts['discovered'], 'unique IDs discovered')}
  {card(counts['excluded'], 'excluded (window or labels)')}
  {card(counts['not_candidates'], 'never sent to OpenAI')}
  {card(counts['classified'], 'classified')}
  {card(counts['unresolved'] + counts['not_yet_classified'], 'unresolved or pending')}
  {card(f"{usage['total_tokens']:,}", 'tokens used (actual)')}
</div>

<h2>Notes</h2>
<ul class="notes">{"".join(f"<li>{escape(note)}</li>" for note in report['notes'])}</ul>
<p class="muted small">Generated {escape(run['report_generated_at'])} · Unread observed {escape(run['unread_observed_at'] or 'when metadata was read')} · Model {escape(run['model'])}</p>
"""
    return page("Newsletter inventory", body, demo)
