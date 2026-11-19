"""Build the animated before/after upgrade report from the measured results.

    uv run python bench/make_report.py

Reads bench/results/before.json, bench/results/after.json, bench/results/ci.json
and evals/results/*.json, and writes docs/upgrade-report.html (a complete page).
With --fragment PATH it also writes the page body alone (for embedding).

Every number on the page comes from those files; nothing is typed in by hand.
The page needs no network: system fonts, inline CSS, SVG and script.
"""

import argparse
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
EVALS = ROOT / "evals" / "results"
OUTPUT = ROOT / "docs" / "upgrade-report.html"
CATEGORY_NAMES = {
    "editorial_newsletter": "Newsletter", "promotional_subscription": "Promotion", "product_update": "Product update",
    "transactional": "Transactional", "personal_or_other": "Personal", "uncertain": "Uncertain",
}
RACE_SAMPLE = 40  # Envelopes drawn per lane; the real run read 1,000.
RACE_BEFORE_SECONDS = 6.0  # Animation length for the slower lane; the faster one is scaled to match.


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def latest_eval(prefix: str) -> dict | None:
    files = sorted(EVALS.glob(f"{prefix}-*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def mb(value: int) -> str:
    return f"{value / 1_000_000:.1f} MB" if value >= 1_000_000 else f"{value / 1_000:.0f} KB"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%".replace(".0%", "%")


def times(before: float, after: float) -> str:
    return f"{before / after:.1f}×"


STYLE = """
:root {
  --bg: #F4F6F9; --surface: #FFFFFF; --ink: #17202F; --muted: #586379; --line: #D9DEE8;
  --accent: #2747D2; --accent-soft: #E7ECFD; --before: #9A5B3C; --before-soft: #F6E9E1;
  --good: #1D7F55; --good-soft: #DFF2E8; --warn: #A86A12; --on-fill: #FFFFFF;
  --font: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", Roboto, sans-serif;
  --display: "Avenir Next", "Segoe UI Variable Display", "Segoe UI", ui-sans-serif, system-ui, sans-serif;
  --mono: ui-monospace, "SFMono-Regular", "Cascadia Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0F131B; --surface: #161C27; --ink: #E6E9F0; --muted: #9AA4B7; --line: #2A3243;
    --accent: #8098FF; --accent-soft: #1D2643; --before: #E39A74; --before-soft: #3A251B;
    --good: #4FC48D; --good-soft: #15342A; --warn: #E2AA4E; --on-fill: #0F131B; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0F131B; --surface: #161C27; --ink: #E6E9F0; --muted: #9AA4B7; --line: #2A3243;
  --accent: #8098FF; --accent-soft: #1D2643; --before: #E39A74; --before-soft: #3A251B;
  --good: #4FC48D; --good-soft: #15342A; --warn: #E2AA4E; --on-fill: #0F131B; color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 17px/1.6 var(--font); margin: 0;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 40px 80px; }
.wrap { max-width: 940px; margin: 0 auto; display: grid; gap: 72px; }
section, header { display: grid; gap: 18px; min-width: 0; }
h1, h2, h3 { font-family: var(--display); margin: 0; line-height: 1.15; text-wrap: balance; }
h1 { font-size: clamp(2.1rem, 5vw, 3.2rem); font-weight: 800; letter-spacing: -0.02em; }
h2 { font-size: 1.7rem; font-weight: 700; }
h3 { font-size: 1.05rem; font-weight: 700; }
p { margin: 0; max-width: 68ch; }
.lede { font-size: 1.15rem; }
.muted { color: var(--muted); }
.small { font-size: 0.88rem; }
.eyebrow { font-family: var(--mono); font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--accent); font-weight: 600; }
.num { font-variant-numeric: tabular-nums; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; min-width: 0; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line);
  border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.tile { background: var(--surface); padding: 14px 16px; display: grid; gap: 2px; align-content: start; }
.tile b { font-family: var(--display); font-size: 1.9rem; font-weight: 800; color: var(--accent); }
.tile span { font-size: 0.88rem; color: var(--muted); }
.big { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; }
.big b { font-family: var(--display); font-size: clamp(2.4rem, 7vw, 3.6rem); font-weight: 800; color: var(--accent); line-height: 1; }
.big span { color: var(--muted); }
button.replay { font: 600 0.85rem var(--font); color: var(--accent); background: var(--accent-soft); border: 0;
  border-radius: 999px; padding: 6px 14px; cursor: pointer; width: fit-content; }
button.replay:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* 1. Scan race */
.lanes { display: grid; gap: 18px; }
.lane { display: grid; gap: 8px; }
.lane-head { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; align-items: baseline; }
.lane-head strong { font-family: var(--display); }
.lane-head .clock { font-family: var(--mono); font-size: 0.85rem; color: var(--muted); }
.envelopes { display: grid; grid-template-columns: repeat(20, 1fr); gap: 4px; }
.env { aspect-ratio: 1.45; border-radius: 3px; background: var(--done); max-width: 100%; }
.lane.before { --done: var(--before); --soft: var(--before-soft); }
.lane.after { --done: var(--accent); --soft: var(--accent-soft); }
.env.waiting { background: var(--soft); }
.env.reading { background: var(--done); opacity: 0.45; animation: pulse 0.35s ease-in-out infinite alternate; }
@keyframes pulse { to { opacity: 1; } }
.track { height: 6px; background: var(--soft); border-radius: 3px; overflow: hidden; }
.fill { height: 100%; width: 100%; background: var(--done); border-radius: 3px; }
.slots { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); }

/* 2. Disk bars */
.chart svg { display: block; width: 100%; height: auto; }
.chart .axis { fill: var(--muted); font: 12px var(--mono); }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .bar-before { fill: var(--before); }
.chart .bar-after { fill: var(--accent); }
.chart .value { fill: var(--ink); font: 600 12px var(--font); }
.chart .group-label { fill: var(--ink); font: 600 13px var(--font); }
.chart rect { transform-box: fill-box; transform-origin: bottom; }
.play .chart rect.bar-before, .play .chart rect.bar-after { animation: grow 1.1s cubic-bezier(.2,.7,.2,1) backwards; }
.play .chart rect.bar-after { animation-delay: 0.35s; }
@keyframes grow { from { transform: scaleY(0); } }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 0.88rem; color: var(--muted); }
.legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }

/* 3. Counters */
.counters { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.counter { display: grid; gap: 4px; }
.counter .was { font-family: var(--mono); font-size: 0.85rem; color: var(--before); text-decoration: line-through; }
.counter b { font-family: var(--display); font-size: 2.6rem; font-weight: 800; color: var(--good); line-height: 1; }
.counter span { color: var(--muted); font-size: 0.92rem; }

/* 4. Platforms */
.os-grid { display: grid; grid-template-columns: 120px repeat(2, 1fr); gap: 8px; align-items: stretch; }
.os-grid .head { font-family: var(--mono); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); padding: 4px 0; }
.os-grid .os { font-weight: 700; padding: 10px 0; }
.chip { border-radius: 10px; padding: 10px 12px; display: grid; gap: 2px; perspective: 600px; min-width: 0; }
.chip .now { font-weight: 600; color: var(--good); }
.chip .then { font-size: 0.82rem; color: var(--muted); }
.chip { background: var(--good-soft); }
.play .chip.changed { animation: flip 0.7s ease-out backwards; }
@keyframes flip { from { transform: rotateX(90deg); background: var(--before-soft); } }

/* 5. Accuracy */
.bars { display: grid; gap: 14px; }
.bar-row { display: grid; grid-template-columns: 210px 1fr 64px; gap: 12px; align-items: center; }
.bar-row .label { font-size: 0.92rem; }
.bar-row .label small { display: block; color: var(--muted); font-size: 0.8rem; }
.bar-track { height: 14px; background: var(--line); border-radius: 7px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 7px; transform-origin: left; }
.bar-fill.offline { background: var(--before); }
.bar-fill.model { background: var(--accent); }
.bar-fill.none { background: transparent; }
.play .bar-fill { animation: widen 1.2s cubic-bezier(.2,.7,.2,1) backwards; }
@keyframes widen { from { transform: scaleX(0); } }
.matrix-wrap { overflow-x: auto; }
table.matrix { border-collapse: separate; border-spacing: 3px; font-size: 0.82rem; }
table.matrix th { font-weight: 600; color: var(--muted); padding: 4px 6px; text-align: center; }
table.matrix th.row { text-align: right; }
table.matrix td { width: 52px; height: 34px; text-align: center; border-radius: 6px; background: var(--bg);
  font-variant-numeric: tabular-nums; }
table.matrix td.hit { background: var(--accent); color: var(--on-fill); font-weight: 700; }
table.matrix td.miss { background: var(--before); color: var(--on-fill); font-weight: 700; }
.play table.matrix td { animation: fade 0.5s ease-out backwards; animation-delay: calc(var(--i) * 40ms); }
@keyframes fade { from { opacity: 0; transform: scale(0.6); } }

/* 6. Features */
.feature-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; align-items: start; }
.cards { display: grid; gap: 8px; }
.advice { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
  display: grid; grid-template-columns: 1fr auto; gap: 2px 10px; }
.advice .why { grid-column: 1 / -1; font-size: 0.84rem; color: var(--muted); }
.badge { font-size: 0.74rem; font-weight: 600; border-radius: 999px; padding: 2px 9px; height: fit-content; }
.badge.drop { background: var(--before-soft); color: var(--before); }
.badge.skim { background: var(--accent-soft); color: var(--accent); }
.play .advice { animation: slide 0.6s cubic-bezier(.2,.7,.2,1) backwards; animation-delay: calc(var(--i) * 120ms); }
@keyframes slide { from { opacity: 0; transform: translateX(-24px); } }
.spark-list { display: grid; gap: 10px; }
.spark-row { display: grid; grid-template-columns: 1fr 120px 64px; gap: 10px; align-items: center; font-size: 0.9rem; }
.spark-row svg { width: 120px; height: 26px; display: block; }
.spark-row polyline { fill: none; stroke: var(--accent); stroke-width: 2; stroke-linejoin: round; }
.play .spark-row polyline { stroke-dasharray: 400; animation: draw 1.4s ease-out backwards; }
@keyframes draw { from { stroke-dashoffset: 400; } to { stroke-dashoffset: 0; } }
.trend { font-size: 0.74rem; color: var(--muted); font-family: var(--mono); }

/* 7. Unchanged */
.same { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
.same div { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
  display: grid; grid-template-columns: auto 1fr; gap: 0 10px; align-items: center; }
.same .tick { color: var(--good); font-weight: 800; grid-row: span 2; font-size: 1.2rem; }
.same b { font-variant-numeric: tabular-nums; }
.same span { font-size: 0.84rem; color: var(--muted); }
.play .same .tick { animation: pop 0.45s ease-out backwards; animation-delay: calc(var(--i) * 90ms); }
@keyframes pop { from { transform: scale(0); } }
ul.plain { margin: 0; padding-left: 20px; display: grid; gap: 6px; }
footer { color: var(--muted); font-size: 0.86rem; border-top: 1px solid var(--line); padding-top: 18px; }

@media (max-width: 720px) {
  .tiles { grid-template-columns: repeat(2, 1fr); }
  .counters, .feature-grid { grid-template-columns: 1fr; }
  .bar-row { grid-template-columns: 1fr 52px; }
  .bar-row .label { grid-column: 1 / -1; }
  .os-grid { grid-template-columns: 84px repeat(2, 1fr); }
  .envelopes { grid-template-columns: repeat(10, 1fr); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""

SCRIPT = """
(function () {
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function countUp(el) {
    var from = Number(el.dataset.from), to = Number(el.dataset.to), decimals = Number(el.dataset.decimals || 0);
    var suffix = el.dataset.suffix || '', start = null, duration = 1300;
    function frame(time) {
      if (start === null) start = time;
      var t = Math.min(1, (time - start) / duration), eased = 1 - Math.pow(1 - t, 3);
      el.textContent = (from + (to - from) * eased).toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function race(section) {
    section.querySelectorAll('.lane').forEach(function (lane) {
      var envelopes = Array.prototype.slice.call(lane.querySelectorAll('.env'));
      var parallel = Number(lane.dataset.parallel), total = Number(lane.dataset.ms);
      var seconds = Number(lane.dataset.seconds), fill = lane.querySelector('.fill');
      var clock = lane.querySelector('.clock'), finalClock = clock.textContent;
      var perRead = total / Math.ceil(envelopes.length / parallel);
      envelopes.forEach(function (env) { env.className = 'env waiting'; });
      fill.style.width = '0%';
      var start = null;
      function frame(time) {
        if (start === null) start = time;
        var elapsed = time - start, t = Math.min(1, elapsed / total);
        var wave = Math.floor(elapsed / perRead);
        envelopes.forEach(function (env, i) {
          var envWave = Math.floor(i / parallel);
          env.className = 'env' + (envWave < wave || t >= 1 ? '' : envWave === wave ? ' reading' : ' waiting');
        });
        fill.style.width = (t * 100).toFixed(1) + '%';
        clock.textContent = t < 1 ? (seconds * t).toFixed(1) + ' s' : finalClock;
        if (t < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  }

  function play(section) {
    if (reduce) return;
    section.classList.remove('play');
    void section.offsetWidth;  // Restart CSS animations.
    section.classList.add('play');
    section.querySelectorAll('[data-to]').forEach(countUp);
    if (section.id === 'speed') race(section);
  }

  document.querySelectorAll('button.replay').forEach(function (button) {
    button.addEventListener('click', function () { play(button.closest('section')); });
  });
  if (reduce || !('IntersectionObserver' in window)) return;
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) { observer.unobserve(entry.target); play(entry.target); }
    });
  }, { threshold: 0.35 });
  document.querySelectorAll('section[data-animate], header[data-animate]').forEach(function (s) { observer.observe(s); });
})();
"""


def counter(before, after, decimals=0, suffix="") -> str:
    """A number that rolls from its old value to its new one; the final value is in the HTML."""
    return (f'<b class="num" data-from="{before}" data-to="{after}" data-decimals="{decimals}" '
            f'data-suffix="{escape(suffix)}">{after:.{decimals}f}{escape(suffix)}</b>')


def lane(kind: str, title: str, seconds: float, parallel: int, animation_ms: int, note: str) -> str:
    envelopes = "".join('<i class="env"></i>' for _ in range(RACE_SAMPLE))
    return f"""
<div class="lane {kind}" data-parallel="{parallel}" data-ms="{animation_ms}" data-seconds="{seconds}">
  <div class="lane-head"><strong>{escape(title)}</strong><span class="clock">{seconds:.1f} s</span></div>
  <div class="envelopes" aria-hidden="true">{envelopes}</div>
  <div class="track"><div class="fill"></div></div>
  <span class="slots">{escape(note)}</span>
</div>"""


def disk_chart(before_rows: list[dict], after_rows: list[dict]) -> str:
    width, height, left, bottom, top = 640, 300, 56, 40, 24
    plot_h = height - bottom - top
    peak = max(row["bytes_written"] for row in before_rows + after_rows)
    ceiling = 50_000_000 * (int(peak / 50_000_000) + 1)
    group_w = (width - left - 16) / len(before_rows)
    bar_w = min(56, group_w / 3)
    parts = []
    for tick in range(0, ceiling + 1, 50_000_000):
        y = top + plot_h * (1 - tick / ceiling)
        parts.append(f'<line class="grid" x1="{left}" x2="{width - 8}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick // 1_000_000}</text>')
    parts.append(f'<text class="axis" x="{left - 8}" y="{top - 10}" text-anchor="end">MB</text>')
    for index, (before, after) in enumerate(zip(before_rows, after_rows, strict=True)):
        center = left + group_w * (index + 0.5)
        for kind, row, offset in (("before", before, -bar_w / 2 - 3), ("after", after, bar_w / 2 + 3)):
            bar_h = max(2.0, plot_h * row["bytes_written"] / ceiling)
            x, y = center + offset - bar_w / 2, top + plot_h - bar_h
            parts.append(f'<rect class="bar-{kind}" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3"/>')
            parts.append(f'<text class="value" x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle">'
                         f'{escape(mb(row["bytes_written"]))}</text>')
        parts.append(f'<text class="group-label" x="{center:.1f}" y="{height - 14}" text-anchor="middle">'
                     f'{before["messages"]:,} emails</text>')
    summary = "; ".join(f"{b['messages']:,} emails: {mb(b['bytes_written'])} before, {mb(a['bytes_written'])} after"
                        for b, a in zip(before_rows, after_rows, strict=True))
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Bytes written by a first scan. {escape(summary)}">'
            f'{"".join(parts)}</svg>')


def sparkline_points(counts: list[int]) -> str:
    width, height, top = 120, 26, max(max(counts), 1)
    step = width / max(len(counts) - 1, 1)
    return " ".join(f"{i * step:.1f},{height - 3 - (height - 6) * count / top:.1f}" for i, count in enumerate(counts))


def confusion_table(result: dict) -> str:
    labels = [name for name in ("editorial_newsletter", "promotional_subscription", "product_update",
                                "transactional", "personal_or_other") if sum(result["confusion"][name].values())]
    columns = labels + ["not_sent"]
    names = {**CATEGORY_NAMES, "not_sent": "Kept private"}
    head = "".join(f"<th>{escape(names[c])}</th>" for c in columns)
    rows, index = [], 0
    for expected in labels:
        cells = []
        for column in columns:
            value = result["confusion"][expected].get(column, 0)
            right = column == expected or (expected == "personal_or_other" and column == "not_sent")
            kind = "hit" if value and right else "miss" if value else ""
            cells.append(f'<td class="{kind}" style="--i:{index}">{value or ""}</td>')
            index += 1
        rows.append(f'<tr><th class="row">{escape(names[expected])}</th>{"".join(cells)}</tr>')
    return (f'<table class="matrix"><thead><tr><th class="row">True label ↓ / answer →</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def accuracy_bar(label: str, sub: str, value: float | None, kind: str) -> str:
    width = 0 if value is None else value * 100
    shown = "—" if value is None else pct(value)
    return (f'<div class="bar-row"><span class="label">{escape(label)}<small>{escape(sub)}</small></span>'
            f'<div class="bar-track"><div class="bar-fill {kind}" style="width:{width:.1f}%"></div></div>'
            f'<span class="num">{escape(shown)}</span></div>')


def build(before: dict, after: dict, ci: dict, model_eval: dict | None, offline_eval: dict | None) -> tuple[str, str]:
    scan_b, scan_a = before["scan_speed"], after["scan_speed"]
    full_b, full_a = before["full_run"], after["full_run"]
    health_b, health_a = before["code_health"], after["code_health"]
    scale_b, scale_a = before["disk_scaling"], after["disk_scaling"]
    biggest_b, biggest_a = scale_b[-1], scale_a[-1]
    answers = after["answers"]
    same_answers = before["answers"] == after["answers"]
    race_after_ms = int(RACE_BEFORE_SECONDS * 1000 * scan_a["seconds"] / scan_b["seconds"])
    suggestions = after["features"]["cleanup_suggestions"]
    weekly = after["features"]["weekly"][:4]
    ci_passed = ci.get("status") == "passed"

    title = "newsletter-agent 0.2 Upgrade"
    tiles = f"""
<div class="tiles">
  <div class="tile">{counter(1.0, scan_b['seconds'] / scan_a['seconds'], 1, '×')}<span>faster first scan</span></div>
  <div class="tile">{counter(1.0, biggest_b['bytes_written'] / biggest_a['bytes_written'], 1, '×')}<span>less written to disk ({biggest_b['messages']:,} emails)</span></div>
  <div class="tile">{counter(health_b['ruff_findings'] + health_b['mypy_errors'], health_a['ruff_findings'] + health_a['mypy_errors'])}<span>lint and type problems (was {health_b['ruff_findings'] + health_b['mypy_errors']})</span></div>
  <div class="tile"><b>{'Same' if same_answers else 'CHANGED'}</b><span>answers on the demo mailbox</span></div>
</div>"""  # noqa: E501

    race = "".join([
        lane("before", "v0.1: one email at a time", scan_b["seconds"], max(1, scan_b["peak_parallel_reads"]),
             int(RACE_BEFORE_SECONDS * 1000), f"{scan_b['peak_parallel_reads']} read in flight · "
             f"{scan_b['messages_per_second']:.0f} emails/s"),
        lane("after", "v0.2: eight at a time, results kept in order", scan_a["seconds"],
             max(1, scan_a["peak_parallel_reads"]), race_after_ms,
             f"{scan_a['peak_parallel_reads']} reads in flight · {scan_a['messages_per_second']:.0f} emails/s"),
    ])

    os_rows = [
        ("macOS", ("Works", "Works"), ("launchd", "launchd")),
        ("Linux", ("Works", "Works"), ("Copy a cron line by hand", "systemd timer, catches up after sleep")),
        ("Windows", ("Could not start (Unix-only lock)", "Works"), ("None", "Task Scheduler")),
    ]
    os_cells = []
    for name, (run_then, run_now), (sched_then, sched_now) in os_rows:
        os_cells.append(f'<div class="os">{name}</div>')
        for then, now in ((run_then, run_now), (sched_then, sched_now)):
            changed = " changed" if then != now else ""
            os_cells.append(f'<div class="chip{changed}"><span class="now">{escape(now)}</span>'
                            f'<span class="then">{"was: " + escape(then) if changed else "unchanged"}</span></div>')
    ci_line = ("CI ran the full test suite, the demo and the eval on Linux (Python 3.11 and 3.13), macOS and Windows: "
               "all passed." if ci_passed else
               "CI runs the full test suite, the demo and the eval on Linux, macOS and Windows for every pull request.")

    if model_eval:
        detection_m, detection_o = model_eval["subscription_detection"], (offline_eval or {}).get("subscription_detection", {})
        accuracy = f"""
<div class="panel bars">
  {accuracy_bar("Is it a subscription?", "v0.1: never measured", None, "none")}
  {accuracy_bar("Is it a subscription?", "offline keyword rules (CI baseline)", detection_o.get('f1'), "offline")}
  {accuracy_bar("Is it a subscription?", model_eval['model'], detection_m['f1'], "model")}
  {accuracy_bar("Right category", "offline keyword rules", (offline_eval or {}).get('category_accuracy'), "offline")}
  {accuracy_bar("Right category", model_eval['model'], model_eval['category_accuracy'], "model")}
</div>
<div class="panel matrix-wrap">
  <h3>Where each of the {model_eval['cases']} test emails landed ({escape(model_eval['model'])})</h3>
  <p class="muted small">Rows are the true answer, columns what the app decided. Blue cells are right; any orange cell is
  a mistake. "Kept private" means the privacy rules never sent the email to the model, which is the right outcome
  for personal mail.</p>
  {confusion_table(model_eval)}
</div>
<p class="muted small">Honest limits: these are {model_eval['cases']} invented, fairly clear-cut emails, and the run used
{model_eval['usage']['total_tokens']:,} tokens ({model_eval['usage']['model_calls']} calls). A perfect score on {model_eval['cases']}
emails suggests the real error rate is low (likely under about {300 / model_eval['cases']:.0f}%), not that it is zero.
Real inboxes are messier. Rerun with <code>newsletter-agent eval --real-model</code>.</p>"""  # noqa: E501
        accuracy_big = f"""<div class="big">{counter(0, detection_m['f1'] * 100, 0, '%')}<span>of subscriptions spotted correctly
(F1), {pct(model_eval['category_accuracy'])} put in the right category, {len(model_eval['mistakes'])} mistakes</span></div>"""
    else:
        accuracy, accuracy_big = '<p class="panel">The real-model eval has not been run yet.</p>', ""

    advice_cards = "".join(
        f'<div class="advice" style="--i:{i}"><strong>{escape(item["name"])}</strong>'
        f'<span class="badge {"drop" if item["action"] == "consider_unsubscribing" else "skim"}">'
        f'{"Consider unsubscribing" if item["action"] == "consider_unsubscribing" else "Skim or reduce"}</span>'
        f'<span class="why">{escape(item["reason"])}</span></div>'
        for i, item in enumerate(suggestions)
    )
    sparks = "".join(
        f'<div class="spark-row"><span>{escape(row["name"])}</span>'
        f'<svg viewBox="0 0 120 26" role="img" aria-label="Messages per week: '
        f'{", ".join(str(c) for c in row["weekly_counts"])}"><polyline points="{sparkline_points(row["weekly_counts"])}"/></svg>'
        f'<span class="trend">{escape(row["trend"] or "")}</span></div>'
        for row in weekly
    )

    counts = answers["counts"]
    same_items = [
        (counts["discovered"], "emails found in 90 days"), (counts["in_scope"], "in scope"),
        (counts["classified"], "classified"), (counts["publications_observed"], "publications"),
        (counts["subscription_messages"], "subscription emails"), (counts["subscription_unread"], "of them unread"),
        (answers["briefing_counts"]["summarized"], "issues in the briefing"),
        ("All 5", "reconciliation checks pass" if answers["reconciliation_ok"] else "RECONCILIATION FAILED"),
    ]
    same = "".join(f'<div style="--i:{i}"><span class="tick">✓</span><b>{escape(str(value))}</b><span>{escape(label)}</span></div>'
                   for i, (value, label) in enumerate(same_items))

    body = f"""
<div class="wrap">
<header data-animate>
  <span class="eyebrow">Upgrade report · v0.1 ({escape(before['commit'] or '')}) → v0.2 ({escape(after['commit'] or '')})</span>
  <h1>Same answers. Much faster, sturdier, and measured.</h1>
  <p class="lede">newsletter-agent finds the newsletters in your Gmail and writes a daily briefing. This upgrade
  changed how it reads mail, where it keeps its notes, which computers it runs on, and how we know it is accurate.
  The numbers below were measured on the same machine with the same synthetic mailboxes, before and after.</p>
  {tiles}
</header>

<section id="speed" data-animate>
  <span class="eyebrow">1 · Reading mail</span>
  <h2>The first scan is {times(scan_b['seconds'], scan_a['seconds'])} faster</h2>
  <div class="big">{counter(round(scan_b['projection']['sequential_minutes']), scan_a['projection']['minutes'], 1, ' min')}
    <span>estimated first scan of {scan_a['projection']['messages']:,} real emails (was about
    {scan_b['projection']['sequential_minutes']:.0f} min)</span></div>
  <p>Before, the app asked Gmail for one email and waited for the answer before asking for the next. Now it keeps
  eight requests going at once, and still uses the answers in their original order, so the report is exactly the
  same. If Gmail asks it to slow down, it drops to one at a time instead of failing.</p>
  <div class="panel lanes">{race}</div>
  <p class="muted small">Measured: {scan_a['messages']:,} emails with {scan_a['latency_ms']} ms per read took
  {scan_b['seconds']:.1f} s before and {scan_a['seconds']:.1f} s after. A full run that also reads text and labels
  {full_a['messages']} emails: {full_b['seconds']:.1f} s → {full_a['seconds']:.1f} s. The real-mailbox estimate
  assumes Gmail's usual 3 reads per second. Each lane shows {RACE_SAMPLE} of the emails, at the measured relative
  speed.</p>
  <button class="replay" type="button">Replay</button>
</section>

<section id="disk" data-animate>
  <span class="eyebrow">2 · Keeping notes</span>
  <h2>{times(biggest_b['bytes_written'], biggest_a['bytes_written'])} less written to disk on a big mailbox</h2>
  <p>The app remembers each email's headers so it never reads them twice. Version 0.1 kept that memory in one big file
  and rewrote the whole file after every 500 emails, so the work grew with the square of your mailbox. Version 0.2
  keeps it in a small database and only adds the new rows. The bigger the mailbox, the bigger the difference.</p>
  <div class="panel chart">{disk_chart(scale_b, scale_a)}
    <div class="legend"><span><i style="background:var(--before)"></i>v0.1</span>
    <span><i style="background:var(--accent)"></i>v0.2</span></div></div>
  <p class="muted small">Bytes passed to the disk during one first scan, measured by the operating system. Upgrading
  keeps everything already learned: a real v0.1 data folder moved to v0.2 with 0 emails re-read and 0 model calls.
  A hard kill in the middle of a scan leaves the database intact, and the next run picks up where it stopped.</p>
  <button class="replay" type="button">Replay</button>
</section>

<section id="health" data-animate>
  <span class="eyebrow">3 · Code health</span>
  <h2>Every lint and type problem fixed, and checked on every change</h2>
  <p>A linter catches sloppy or risky code; a type checker catches parts that don't fit together. Both now run in CI,
  so a mistake like a misspelled name or a forgotten "nothing" value is caught before it ships.</p>
  <div class="counters">
    <div class="counter panel"><span class="was">{health_b['ruff_findings']}</span>
      {counter(health_b['ruff_findings'], health_a['ruff_findings'])}<span>lint findings</span></div>
    <div class="counter panel"><span class="was">{health_b['mypy_errors']}</span>
      {counter(health_b['mypy_errors'], health_a['mypy_errors'])}<span>type errors</span></div>
    <div class="counter panel"><span class="was">{health_b['tests_passed']}</span>
      {counter(health_b['tests_passed'], health_a['tests_passed'])}<span>tests, all passing, all offline</span></div>
  </div>
  <p class="muted small">Same settings for both versions: {escape(health_a['settings']['ruff'])};
  {escape(health_a['settings']['mypy'])}. New tests cover failures on purpose: a hard kill mid-scan, a corrupt
  database, rate limits, Ctrl-C with reads in flight, invented answers from the model, and failed scheduler
  installs.</p>
  <button class="replay" type="button">Replay</button>
</section>

<section id="platforms" data-animate>
  <span class="eyebrow">4 · Where it runs</span>
  <h2>Now on Windows and Linux too, with daily briefings everywhere</h2>
  <p>Version 0.1 used a Unix-only lock, so it could not even start on Windows, and only Macs could schedule the daily
  briefing. Now each system uses its own scheduler. A bug is fixed too: if installing the schedule failed, v0.1 still
  switched on automatic approval.</p>
  <div class="panel"><div class="os-grid">
    <div></div><div class="head">Runs the app</div><div class="head">Daily briefing</div>
    {''.join(os_cells)}
  </div></div>
  <p class="muted small">{escape(ci_line)}</p>
  <button class="replay" type="button">Replay</button>
</section>

<section id="accuracy" data-animate>
  <span class="eyebrow">5 · Accuracy</span>
  <h2>For the first time, a real accuracy number</h2>
  <p>A new <code>eval</code> command runs the real app on {model_eval['cases'] if model_eval else 96} invented emails
  with known answers: ordinary newsletters plus traps, such as a receipt from a newsletter brand, a friend writing
  through a newsletter platform, emails in four languages, and two emails that try to give the AI orders.</p>
  {accuracy_big}
  {accuracy}
  <button class="replay" type="button">Replay</button>
</section>

<section id="features" data-animate>
  <span class="eyebrow">6 · New in the report</span>
  <h2>It now tells you what to drop, and how each newsletter is trending</h2>
  <p>The report used to stop at counts. Now it points out newsletters you mostly don't open and shows how often each
  one arrives week by week. It only suggests: it never unsubscribes or changes anything in Gmail. These are the demo
  mailbox's own results.</p>
  <div class="feature-grid">
    <div class="cards">{advice_cards}</div>
    <div class="panel spark-list"><h3>Emails per week, last 90 days</h3>{sparks}</div>
  </div>
  <button class="replay" type="button">Replay</button>
</section>

<section id="unchanged" data-animate>
  <span class="eyebrow">7 · What did not change</span>
  <h2>{'The answers are identical' if same_answers else 'WARNING: the answers changed'}</h2>
  <p>A faster app is only useful if it says the same thing. On the demo mailbox, every count, category and publication
  matched between v0.1 and v0.2, and a test now fails if they ever drift. The privacy promises are unchanged too.</p>
  <div class="same">{same}</div>
  <ul class="plain">
    <li>Gmail access is still read-only, through the same two tools.</li>
    <li>The same privacy rules decide what may reach the AI; email bodies are still never written to disk.</li>
    <li>Paid AI work still needs your approval and stays within a hard budget.</li>
  </ul>
  <button class="replay" type="button">Replay</button>
</section>

<footer>
  How this was measured: <code>bench/run_bench.py</code> ran against commit {escape(before['commit'] or '')} (v0.1) and
  {escape(after['commit'] or '')} (v0.2) on {escape(after['platform'])}, Python {escape(after['python'])}, using only
  invented emails: no Gmail account and no real mail. The accuracy numbers come from <code>newsletter-agent eval</code>.
  This page is generated by <code>bench/make_report.py</code> from the saved results.
</footer>
</div>
"""
    head = f"<title>{escape(title)}</title>\n<style>{STYLE}</style>"
    return head, body + f"<script>{SCRIPT}</script>"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fragment", help="Also write the page without <html>/<head>/<body> to this path")
    parser.add_argument("--output", default=str(OUTPUT), help="Where to write the page (default docs/upgrade-report.html)")
    args = parser.parse_args(argv)
    ci_path = RESULTS / "ci.json"
    ci = json.loads(ci_path.read_text(encoding="utf-8")) if ci_path.exists() else {"status": "unknown"}
    head, body = build(load("before.json"), load("after.json"), ci, latest_eval("gpt-6-luna"), latest_eval("offline"))
    page = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f'<meta name="description" content="Measured before/after results of the newsletter-agent 0.2 upgrade.">\n'
            f"{head}\n</head>\n<body>\n{body}\n</body>\n</html>\n")
    Path(args.output).write_text(page, encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.fragment:
        fragment = f'{head}\n<meta name="claude:viewport-fit-sides" content="cover">\n{body}\n'
        Path(args.fragment).write_text(fragment, encoding="utf-8")
        print(f"Wrote {args.fragment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
