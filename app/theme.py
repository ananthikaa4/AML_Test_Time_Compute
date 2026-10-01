"""Visual layer: a small, calm CSS payload plus a few HTML builders.

Design intent: a quiet editorial look on warm paper, one accent colour, generous
whitespace, tabular numerals. Everything stays legible when projected. Motion is limited
to short fades, so nothing distracts during a live demo.
"""
from __future__ import annotations

import html

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --paper:#FAF8F4; --surface:#FFFFFF; --surface-2:#F4F1EB;
  --ink:#16171B; --ink-2:#45474F; --muted:#787A84; --line:#E6E1D7; --line-2:#F0ECE3;
  --accent:#0F5A4A; --accent-soft:#E7F1ED;
  --good:#17733F; --good-soft:#E8F3EB;
  --warn:#9A5B10; --warn-soft:#FBF1E3;
  --bad:#A42A22; --bad-soft:#FAEBE9;
  --sans:'Inter',system-ui,-apple-system,sans-serif;
  --serif:'Source Serif 4',Georgia,serif;
  --mono:'IBM Plex Mono',ui-monospace,monospace;
}

/* hide Streamlit chrome */
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"]{display:none!important}
[data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp{background:var(--paper)!important}
[data-testid="stMainBlockContainer"], .stMainBlockContainer, .block-container{
  padding:34px clamp(18px,4vw,46px) 70px!important; max-width:1140px!important}
[data-testid="stVerticalBlock"]{gap:.9rem!important}
html, body, .stApp, [class*="st-"]{font-family:var(--sans); color:var(--ink);
  -webkit-font-smoothing:antialiased}
::selection{background:#D8E8E2}
a{color:var(--accent)!important}

/* ---------- typography ---------- */
.t-kicker{font-family:var(--mono); font-size:11px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted)}
h1.t-h1{font-family:var(--serif); font-weight:600; font-size:clamp(30px,4.2vw,48px); line-height:1.06;
  letter-spacing:-.02em; margin:12px 0 0; color:var(--ink)}
h1.t-h1 em{font-style:italic; color:var(--accent)}
.t-lede{font-size:16.5px; line-height:1.62; color:var(--ink-2); max-width:70ch; margin:16px 0 0}
.t-lede b{color:var(--ink); font-weight:600}
h2.t-h2{font-family:var(--serif); font-weight:600; font-size:25px; letter-spacing:-.015em;
  margin:4px 0 2px; color:var(--ink)}
.t-sub{font-size:14.5px; line-height:1.6; color:var(--muted); max-width:78ch; margin:0 0 4px}
.t-rule{height:1px; background:var(--line); margin:26px 0 22px; border:0}

/* ---------- cards ---------- */
.t-card{background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:22px 24px;
  box-shadow:0 1px 2px rgba(22,23,27,.03); animation:t-in .45s ease both}
.t-card.accent{border-color:#CFE2DB; background:linear-gradient(180deg,#FBFDFC,#FFFFFF)}
.t-card.flat{box-shadow:none; background:var(--surface-2); border-color:var(--line-2)}
.t-cardtitle{font-family:var(--mono); font-size:10.5px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); margin:0 0 16px}
@keyframes t-in{from{opacity:0; transform:translateY(6px)} to{opacity:1; transform:none}}
@keyframes t-grow{from{transform:scaleX(0)} to{transform:scaleX(1)}}

/* ---------- metric strip ---------- */
.t-metrics{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:14px; overflow:hidden}
.t-metric{background:var(--surface); padding:18px 20px}
.t-metric .v{font-family:var(--serif); font-size:30px; font-weight:600; letter-spacing:-.02em;
  line-height:1; font-variant-numeric:tabular-nums; color:var(--ink)}
.t-metric .v small{font-size:.5em; color:var(--muted); margin-left:3px; font-weight:400}
.t-metric .k{font-size:12px; color:var(--muted); margin-top:9px; line-height:1.4}
.t-metric.hl .v{color:var(--accent)}

/* ---------- step trace ---------- */
.t-steps{display:grid; gap:0}
.t-step{display:grid; grid-template-columns:30px 1fr; gap:16px; padding:17px 0;
  border-bottom:1px solid var(--line-2)}
.t-step:last-child{border-bottom:0; padding-bottom:2px}
.t-step .n{width:25px; height:25px; border-radius:50%; background:var(--accent-soft); color:var(--accent);
  font-family:var(--mono); font-size:11px; display:flex; align-items:center; justify-content:center;
  font-weight:500; margin-top:1px}
.t-step .ttl{font-size:13px; font-weight:600; color:var(--ink); letter-spacing:-.005em}
.t-step .bdy{font-size:14px; color:var(--ink-2); line-height:1.55; margin-top:5px}
.t-step .bdy b{color:var(--ink); font-weight:600; font-variant-numeric:tabular-nums}
.t-step .q{font-family:var(--serif); font-size:16px; color:var(--ink); line-height:1.5;
  border-left:2px solid var(--line); padding-left:14px; margin-top:7px}

/* ---------- verdict ---------- */
.t-verdict{display:flex; align-items:flex-end; justify-content:space-between; gap:22px; flex-wrap:wrap}
.t-prov{font-family:var(--serif); font-size:clamp(28px,4vw,40px); font-weight:600; letter-spacing:-.02em;
  line-height:1.05; margin:0; color:var(--ink)}
.t-provsub{font-size:14.5px; color:var(--muted); margin-top:7px}
.t-confbox{text-align:right}
.t-confbox .v{font-family:var(--serif); font-size:32px; font-weight:600; color:var(--accent);
  font-variant-numeric:tabular-nums; line-height:1}
.t-confbox .k{font-size:11.5px; color:var(--muted); margin-top:6px}

/* ---------- chips ---------- */
.t-chips{display:flex; gap:7px; flex-wrap:wrap; margin-top:16px}
.t-chip{font-size:12px; padding:5px 11px; border-radius:7px; background:var(--surface-2);
  border:1px solid var(--line-2); color:var(--ink-2); white-space:nowrap}
.t-chip b{font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums}
.t-chip.good{background:var(--good-soft); border-color:#CBE3D4; color:var(--good)}
.t-chip.bad{background:var(--bad-soft); border-color:#EED5D2; color:var(--bad)}
.t-chip.warn{background:var(--warn-soft); border-color:#EEDCC2; color:var(--warn)}
.t-chip.easy{background:var(--good-soft); border-color:#CBE3D4; color:var(--good)}
.t-chip.medium{background:var(--warn-soft); border-color:#EEDCC2; color:var(--warn)}
.t-chip.hard{background:var(--bad-soft); border-color:#EED5D2; color:var(--bad)}

/* ---------- banner ---------- */
.t-banner{border-radius:12px; padding:15px 18px; font-size:14.5px; line-height:1.55;
  display:flex; gap:12px; align-items:flex-start; animation:t-in .4s ease both}
.t-banner .ic{font-size:15px; line-height:1.4}
.t-banner.good{background:var(--good-soft); border:1px solid #CBE3D4; color:#114D2C}
.t-banner.bad{background:var(--warn-soft); border:1px solid #EEDCC2; color:#6E4209}
.t-banner.info{background:var(--surface-2); border:1px solid var(--line-2); color:var(--ink-2)}
.t-banner b{font-weight:600}

/* ---------- ranked list ---------- */
.t-rank{display:grid; gap:0}
.t-item{display:grid; grid-template-columns:20px 1fr 54px; gap:13px; align-items:center;
  padding:11px 0; border-bottom:1px solid var(--line-2)}
.t-item:last-child{border-bottom:0; padding-bottom:0}
.t-item .i{font-family:var(--mono); font-size:11px; color:#A9AAB3}
.t-item .nm{font-size:14.5px; font-weight:600; color:var(--ink); letter-spacing:-.005em}
.t-item .sb{font-size:11.5px; color:var(--muted); margin-top:3px}
.t-item .pc{font-family:var(--mono); font-size:12px; text-align:right; color:var(--ink-2);
  font-variant-numeric:tabular-nums}
.t-item .meter{height:4px; border-radius:4px; background:var(--surface-2); margin-top:7px; overflow:hidden}
.t-item .meter i{display:block; height:100%; border-radius:4px; background:var(--accent);
  transform-origin:left; animation:t-grow .6s ease both}
.t-item.gold .nm{color:var(--good)}
.t-item.gold .meter i{background:var(--good)}

/* ---------- table ---------- */
.t-tbl{width:100%; border-collapse:collapse; font-size:14px}
.t-tbl th{font-family:var(--mono); font-size:10px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); text-align:left; font-weight:400; padding:0 12px 11px 0;
  border-bottom:1px solid var(--line)}
.t-tbl td{padding:13px 12px 13px 0; border-bottom:1px solid var(--line-2); vertical-align:middle}
.t-tbl tr:last-child td{border-bottom:0}
.t-tbl .nm{font-weight:600}
.t-tbl .num{font-family:var(--mono); font-size:12.5px; font-variant-numeric:tabular-nums; color:var(--ink-2)}
.t-tbl .num.strong{color:var(--ink); font-weight:500}
.t-tbl tr.hl td{background:#FBFDFC}
.t-tbl tr.hl .nm{color:var(--accent)}
.t-bar{position:relative; height:7px; border-radius:7px; background:var(--surface-2); min-width:80px}
.t-bar i{position:absolute; left:0; top:0; bottom:0; border-radius:7px; background:var(--accent);
  transform-origin:left; animation:t-grow .7s ease both}
.t-bar.soft i{background:#B9C3BF}

/* ---------- key/value ---------- */
.t-kv{display:grid; gap:0}
.t-kv > div{display:flex; justify-content:space-between; gap:16px; padding:10px 0;
  border-bottom:1px solid var(--line-2); font-size:13.5px}
.t-kv > div:last-child{border-bottom:0}
.t-kv .k{color:var(--muted)}
.t-kv .v{font-weight:600; font-variant-numeric:tabular-nums; text-align:right}

/* ---------- budget ladder ---------- */
.t-ladder{display:grid; grid-template-columns:repeat(3,1fr); gap:9px}
.t-rung{border:1px solid var(--line); border-radius:11px; padding:13px 14px; background:var(--surface)}
.t-rung.on{border-color:var(--accent); background:var(--accent-soft)}
.t-rung .lb{font-family:var(--mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted)}
.t-rung .vl{font-family:var(--serif); font-size:22px; font-weight:600; margin-top:6px;
  font-variant-numeric:tabular-nums}
.t-rung.on .lb, .t-rung.on .vl{color:var(--accent)}
.t-rung .hint{font-size:11px; color:var(--muted); margin-top:3px}

/* ---------- Streamlit widgets ---------- */
[data-baseweb="tab-list"]{gap:4px!important; border-bottom:1px solid var(--line)!important;
  background:transparent!important; margin-bottom:22px!important}
[data-baseweb="tab"]{background:transparent!important; padding:11px 16px!important;
  border-radius:0!important}
[data-baseweb="tab"] p{font-size:13.5px!important; font-weight:500!important; color:var(--muted)!important}
[data-baseweb="tab"][aria-selected="true"] p{color:var(--accent)!important; font-weight:600!important}
[data-baseweb="tab-highlight"]{background:var(--accent)!important; height:2px!important}
[data-baseweb="tab-border"]{display:none!important}

[data-testid="stTextArea"] label p, [data-testid="stSelectbox"] label p, [data-testid="stRadio"] label p,
[data-testid="stTextArea"] label, [data-testid="stSelectbox"] label{font-size:12px!important;
  font-weight:500!important; color:var(--muted)!important}
[data-testid="stTextArea"] textarea{background:var(--surface)!important; color:var(--ink)!important;
  border:1px solid var(--line)!important; border-radius:12px!important; padding:15px 16px!important;
  font-family:var(--serif)!important; font-size:16.5px!important; line-height:1.55!important;
  box-shadow:none!important}
[data-testid="stTextArea"] textarea:focus{border-color:var(--accent)!important;
  box-shadow:0 0 0 3px var(--accent-soft)!important}
[data-testid="stTextArea"] textarea::placeholder{color:#B0B1B9!important}
[data-testid="stTextArea"] div[data-baseweb="base-input"]{background:transparent!important;border:0!important}

.stButton > button{border-radius:10px!important; border:1px solid var(--line)!important;
  background:var(--surface)!important; color:var(--ink)!important; font-size:13.5px!important;
  font-weight:500!important; padding:11px 18px!important; transition:background .18s,border-color .18s}
.stButton > button:hover{border-color:#C9C3B6!important; background:var(--surface-2)!important;
  color:var(--ink)!important}
.stButton > button[kind="primary"]{background:var(--accent)!important; border-color:var(--accent)!important;
  color:#FFFFFF!important; font-weight:600!important}
.stButton > button[kind="primary"]:hover{background:#0B4639!important; border-color:#0B4639!important;
  color:#FFFFFF!important}

[data-baseweb="select"] > div{background:var(--surface)!important; border:1px solid var(--line)!important;
  border-radius:10px!important; font-size:13.5px!important; color:var(--ink)!important}
[data-baseweb="popover"] li{font-size:13px!important}
[data-testid="stRadio"] div[role="radiogroup"]{gap:6px!important; flex-wrap:wrap!important}
[data-testid="stRadio"] label{background:var(--surface)!important; border:1px solid var(--line)!important;
  border-radius:9px!important; padding:8px 13px!important; margin:0!important}
[data-testid="stRadio"] label p{font-size:12.5px!important; color:var(--ink-2)!important}
[data-testid="stRadio"] label:has(input:checked){background:var(--accent-soft)!important;
  border-color:#BFD9D1!important}
[data-testid="stRadio"] label:has(input:checked) p{color:var(--accent)!important; font-weight:600!important}
[data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child{display:none!important}

[data-testid="stExpander"]{border:1px solid var(--line)!important; border-radius:12px!important;
  background:var(--surface)!important}
[data-testid="stExpander"] summary{font-size:13px!important; font-weight:500!important; color:var(--ink-2)!important}
[data-testid="stExpander"] p, [data-testid="stExpander"] li{font-size:13.5px!important; color:var(--ink-2)!important}
[data-testid="stImage"] img{border-radius:10px; border:1px solid var(--line); background:#fff}
[data-testid="stCaptionContainer"] p{font-size:11.5px!important; color:var(--muted)!important}
[data-testid="stSpinner"] p{font-size:12.5px!important; color:var(--accent)!important}
[data-testid="stAlert"]{border-radius:11px!important}
hr{border-color:var(--line)!important}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
@media (max-width:720px){.t-ladder{grid-template-columns:1fr}}
</style>
"""


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def header(kicker: str, title_html: str, lede_html: str) -> str:
    return (f'<div class="t-kicker">{esc(kicker)}</div>'
            f'<h1 class="t-h1">{title_html}</h1><p class="t-lede">{lede_html}</p>')


def h2(title: str, sub: str = "") -> str:
    s = f'<p class="t-sub">{esc(sub)}</p>' if sub else ""
    return f'<h2 class="t-h2">{esc(title)}</h2>{s}'


def metrics(items: list[tuple[str, str, str, bool]]) -> str:
    cells = ""
    for value, unit, key, hl in items:
        u = f"<small>{esc(unit)}</small>" if unit else ""
        cells += (f'<div class="t-metric{" hl" if hl else ""}"><div class="v">{esc(value)}{u}</div>'
                  f'<div class="k">{esc(key)}</div></div>')
    return f'<div class="t-metrics">{cells}</div>'


def card(title: str, body_html: str, variant: str = "") -> str:
    t = f'<p class="t-cardtitle">{esc(title)}</p>' if title else ""
    return f'<div class="t-card {variant}">{t}{body_html}</div>'


def banner(kind: str, icon: str, text_html: str) -> str:
    return f'<div class="t-banner {kind}"><span class="ic">{icon}</span><span>{text_html}</span></div>'


def steps(items: list[tuple[str, str]]) -> str:
    body = ""
    for i, (title, html_body) in enumerate(items, 1):
        body += (f'<div class="t-step"><div class="n">{i}</div><div>'
                 f'<div class="ttl">{esc(title)}</div><div class="bdy">{html_body}</div></div></div>')
    return f'<div class="t-steps">{body}</div>'


def ranked(items, gold: str | None = None) -> str:
    rows = ""
    for i, r in enumerate(items, 1):
        pct = max(0.0, min(1.0, r.prob)) * 100
        is_gold = bool(gold) and r.label == gold
        tag = ' &nbsp;<span style="color:var(--good);font-size:11.5px">correct answer</span>' if is_gold else ""
        seen = (f"{r.train_support} training example{'s' if r.train_support != 1 else ''}"
                if r.train_support else "never seen in training")
        rows += (f'<div class="t-item{" gold" if is_gold else ""}"><div class="i">{i}</div>'
                 f'<div><div class="nm">{esc(r.name)}{tag}</div>'
                 f'<div class="sb">{esc(r.source)} · {esc(seen)}</div>'
                 f'<div class="meter"><i style="width:{pct:.2f}%"></i></div></div>'
                 f'<div class="pc">{pct:.1f}%</div></div>')
    return f'<div class="t-rank">{rows}</div>'


def kv(items: list[tuple[str, str]]) -> str:
    body = "".join(f'<div><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>'
                   for k, v in items)
    return f'<div class="t-kv">{body}</div>'


def chips(items: list[tuple[str, str]]) -> str:
    return '<div class="t-chips">' + "".join(
        f'<span class="t-chip {k}">{v}</span>' for k, v in items) + "</div>"


def ladder(active: str, budgets: dict[str, int], hints: dict[str, str]) -> str:
    out = ""
    for name, n in budgets.items():
        out += (f'<div class="t-rung{" on" if name == active else ""}">'
                f'<div class="lb">{esc(name)}</div><div class="vl">{n} passes</div>'
                f'<div class="hint">{esc(hints.get(name, ""))}</div></div>')
    return f'<div class="t-ladder">{out}</div>'
