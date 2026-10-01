"""Streamlit demo: difficulty-aware adaptive test-time compute for Indian legal reasoning.

Run from the project root:
    .venv/Scripts/python -m streamlit run app/main.py

The headline result this app is built to communicate:

  Task A asks a model to name the provision (Constitution Article or statute Section) that
  governs a legal question, choosing from 1,932 provisions seen in training. The fine-tuned
  InLegalBERT reaches 30.4% exact match (49.7% on Constitution Articles) where chance is
  0.05%. Spending more test-time compute does NOT raise that accuracy, but routing compute
  by predicted difficulty reaches the accuracy of a fixed 16-pass budget using 7.2 passes.
  The useful outcome is therefore a ~55% compute saving at equal accuracy, not an accuracy
  gain -- and the real bottleneck is the label space (statute Sections score 12.4%).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Adaptive compute for Indian legal reasoning",
                   page_icon="§", layout="wide", initial_sidebar_state="collapsed")

from app import engine, theme  # noqa: E402
from app.theme import banner, card, chips, esc, h2, kv, ladder, metrics, ranked, steps  # noqa: E402

BUDGETS = {"easy": 2, "medium": 8, "hard": 16}
STRATEGIES = {
    "adaptive": "Adaptive (router decides)",
    "single": "1 pass",
    "fixed_4": "4 passes",
    "fixed_8": "8 passes",
    "fixed_16": "16 passes",
}

st.markdown(theme.CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- numbers
@st.cache_data(show_spinner=False)
def facts() -> dict:
    seed = engine.read_csv("results/transformer_seed_summary.csv")
    base = engine.read_csv("results/baseline_results.csv")
    tt = engine.read_csv("results/tt_compute_summary.csv")
    sig_tt = engine.read_json("results/tt_compute_significance.json")
    sig_lr = engine.read_json("results/significance_transformer_vs_lr.json")

    s = seed[seed.split == "test"].set_index(["group", "value"])
    b = base[(base.split == "test") & (base.group == "all")].set_index("model")
    t = tt[tt.split == "test"].set_index("strategy")
    return {
        "acc": float(s.loc[("all", "all"), "accuracy_mean"]),
        "acc_sd": float(s.loc[("all", "all"), "accuracy_std"]),
        "top5": float(s.loc[("all", "all"), "top5_mean"]),
        "f1": float(s.loc[("all", "all"), "macro_f1_mean"]),
        "con": float(s.loc[("tier", "constitution"), "accuracy_mean"]),
        "con_top5": float(s.loc[("tier", "constitution"), "top5_mean"]),
        "sta": float(s.loc[("tier", "statute"), "accuracy_mean"]),
        "sta_top5": float(s.loc[("tier", "statute"), "top5_mean"]),
        "lr": float(b.loc["tfidf_lr", "accuracy"]), "lr_top5": float(b.loc["tfidf_lr", "top5"]),
        "lr_f1": float(b.loc["tfidf_lr", "macro_f1"]), "lr_ms": float(b.loc["tfidf_lr", "ms_per_question"]),
        "knn": float(b.loc["tfidf_knn", "accuracy"]), "knn_top5": float(b.loc["tfidf_knn", "top5"]),
        "knn_f1": float(b.loc["tfidf_knn", "macro_f1"]), "knn_ms": float(b.loc["tfidf_knn", "ms_per_question"]),
        "tt": t, "sig_tt": sig_tt, "sig_lr": sig_lr,
    }


with st.spinner("Loading the fine-tuned InLegalBERT checkpoint …"):
    WARM = engine.warm_up()
F = facts()
_, _, ENC, RC = engine.get_transformer()
N_CLASSES = len(ENC)
CHANCE = 1 / N_CLASSES
FRAMES = engine.get_task_a_frames()
LO, HI = engine.get_thresholds()
SAVING = 1 - float(F["tt"].loc["adaptive_confidence", "avg_calls"]) / 16


# --------------------------------------------------------------------------- header
st.markdown(theme.header(
    "Task A · provision identification · research prototype",
    'Which law <em>governs</em> this question?',
    f"A fine-tuned <b>InLegalBERT</b> reads an Indian legal question and names the provision that "
    f"governs it — one of <b>{N_CLASSES:,}</b> Constitution Articles and statute Sections. "
    f"Before answering it scores its own difficulty, and that score decides how much compute the "
    f"question gets: <b>2, 8 or 16</b> forward passes. "
    f"The point of the project is the trade-off, not the raw score."), unsafe_allow_html=True)

st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
st.markdown(metrics([
    (f"{F['acc'] * 100:.1f}%", "", f"Exact-match accuracy on 722 held-out questions "
                                   f"(chance = {CHANCE * 100:.2f}%)", True),
    (f"{F['con'] * 100:.1f}%", "", "On Constitution Articles, the cleaner label tier", False),
    (f"{F['top5'] * 100:.1f}%", "", "The right provision is in the model's top 5", False),
    (f"{SAVING * 100:.0f}%", "", "Less compute than a fixed 16-pass budget, same accuracy", True),
]), unsafe_allow_html=True)

st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
tab_try, tab_result, tab_how = st.tabs(["  Try the model  ", "  What it found  ", "  How it works  "])


# =========================================================================== TRY
with tab_try:
    st.markdown(h2("Ask a question, watch the four steps",
                   "The model only ever sees the question text. No reference answer, no "
                   "chain-of-thought, no retrieval."), unsafe_allow_html=True)

    EX = engine.get_examples(8)
    EX_MAP = {f"{r.prompt[:92]}{'…' if len(r.prompt) > 92 else ''}": r.prompt for r in EX.itertuples()}
    PICK = "— choose a held-out question (answer is known) —"

    if "q" not in st.session_state:
        # a held-out test question, so the very first screen shows a checkable answer
        st.session_state["q"] = ("Can Parliament make laws regarding the conditions of service "
                                 "for officers and servants of the Supreme Court?")

    def _use_example() -> None:
        c = st.session_state.get("ex_pick")
        if c and c != PICK:
            st.session_state["q"] = EX_MAP[c]

    def _shuffle() -> None:
        pool = FRAMES["test"]
        st.session_state["q"] = str(pool.sample(1, random_state=random.randint(0, 10**6)).iloc[0]["prompt"])
        st.session_state["ex_pick"] = PICK

    c1, c2 = st.columns([3, 1], gap="medium")
    with c1:
        st.selectbox("Examples from the held-out test split", [PICK, *EX_MAP],
                     key="ex_pick", on_change=_use_example)
    with c2:
        st.markdown('<div style="height:25px"></div>', unsafe_allow_html=True)
        st.button("Shuffle a test question", on_click=_shuffle, width="stretch")

    question = st.text_area("Legal question", key="q", height=120,
                            placeholder="e.g. Under which section is a bounced cheque prosecuted?")

    c3, c4 = st.columns([3, 1], gap="medium")
    with c3:
        strategy = st.radio("Compute budget", list(STRATEGIES),
                            format_func=lambda k: STRATEGIES[k], horizontal=True, index=0)
    with c4:
        st.markdown('<div style="height:25px"></div>', unsafe_allow_html=True)
        run = st.button("Identify the provision", type="primary", width="stretch")

    if run or "result" not in st.session_state or st.session_state.get("last") != (question, strategy):
        if question.strip():
            with st.spinner("Scoring difficulty, then running the forward passes …"):
                st.session_state["result"] = engine.analyse(question, strategy=strategy, with_baseline=True)
            st.session_state["last"] = (question, strategy)
        else:
            st.warning("Type a question first.")

    A = st.session_state.get("result")
    if A is not None:
        # ---- verdict
        if A.correct is True:
            verdict = banner("good", "✓", f"<b>Correct.</b> This question comes from the held-out test "
                                          f"split, and the model's answer matches the provision cited "
                                          f"in the reference answer.")
        elif A.correct is False:
            gname, gsrc = engine.pretty_label(A.gold)
            rank = A.gold_rank
            where = (f"the model ranked it <b>#{rank}</b> of {N_CLASSES:,}" if rank
                     else f"it is not in the model's top 5 of {N_CLASSES:,}")
            verdict = banner("bad", "→", f"<b>Not the reference answer.</b> The reference cites "
                                         f"<b>{esc(gname)}</b> ({esc(gsrc)}) and {where}.")
        else:
            verdict = banner("info", "·", "This question is not in the dataset, so there is no reference "
                                          "answer to check against. Judge it yourself against the top 5 below.")
        st.markdown(verdict, unsafe_allow_html=True)
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

        st.markdown(card("", f"""<div class="t-verdict">
  <div><p class="t-prov">{esc(A.top.name)}</p><div class="t-provsub">{esc(A.top.source)}</div></div>
  <div class="t-confbox"><div class="v">{A.top.prob * 100:.0f}%</div>
    <div class="k">model confidence</div></div></div>""" + chips([
            (A.bucket, f'difficulty: <b>{esc(A.bucket)}</b>'),
            ("", f'<b>{A.passes}</b> forward pass{"es" if A.passes != 1 else ""}'),
            ("", f'<b>{A.ms_total:.0f}</b> ms'),
            ("", f'choosing from <b>{N_CLASSES:,}</b> provisions'),
        ]), variant="accent"), unsafe_allow_html=True)

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        left, right = st.columns([1.15, 1], gap="medium")

        with left:
            moved = "yes — the extra passes changed the answer" if (
                A.adaptive[0].label != A.deterministic[0].label) else "no — the extra passes confirmed it"
            st.markdown(card("The four steps, for this question", steps([
                ("The model reads the question",
                 f'<div class="q">{esc(A.question[:260])}{"…" if len(A.question) > 260 else ""}</div>'),
                ("One pass scores its own difficulty",
                 f'Top-1 confidence came out at <b>{A.det_confidence * 100:.1f}%</b>. '
                 f'Validation says confidence ≥ <b>{HI * 100:.0f}%</b> is easy and '
                 f'&lt; <b>{LO * 100:.0f}%</b> is hard, so this question is '
                 f'<b>{esc(A.bucket)}</b>.'),
                ("The budget follows from the difficulty",
                 f'<b>{A.passes}</b> pass{"es" if A.passes != 1 else ""} with dropout left on, '
                 f'and the probabilities are averaged. Did it change the answer? {moved}.'),
                ("The highest-scoring provision is the answer",
                 f'<b>{esc(A.top.name)}</b> at <b>{A.top.prob * 100:.1f}%</b>, '
                 f'with runner-up <b>{esc(A.adaptive[1].name)}</b> at '
                 f'<b>{A.adaptive[1].prob * 100:.1f}%</b>.'),
            ])), unsafe_allow_html=True)

        with right:
            st.markdown(card("Its five best guesses", ranked(A.adaptive, gold=A.gold)),
                        unsafe_allow_html=True)
            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            st.markdown(card("Compute router", ladder(A.bucket, BUDGETS, {
                "easy": f"confidence ≥ {HI * 100:.0f}%",
                "medium": f"{LO * 100:.0f}–{HI * 100:.0f}%",
                "hard": f"below {LO * 100:.0f}%",
            }) + '<div style="height:14px"></div>' + kv([
                ("Thresholds set on", "validation split only"),
                ("Latency per pass", f"{A.ms_per_pass:.1f} ms"),
                ("Uncertainty (entropy)", f"{A.entropy:.2f}"),
                ("Lead over runner-up", f"{A.margin * 100:.1f} points"),
            ])), unsafe_allow_html=True)

        if A.baseline:
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            b = A.baseline[0]
            agree = b.label == A.top.label
            note = ("It lands on the <b>same</b> provision as the transformer." if agree else
                    "It lands on a <b>different</b> provision than the transformer.")
            st.markdown(card(
                "Same question, classical baseline (TF-IDF + nearest neighbours)",
                f'<div style="font-size:17px;font-weight:600">{esc(b.name)} '
                f'<span style="color:var(--muted);font-weight:400;font-size:14px">· '
                f'{esc(b.source)} · {b.prob * 100:.0f}% · {A.baseline_ms:.0f} ms</span></div>'
                f'<div style="height:14px"></div>'
                + banner("info", "·", f"{note} Across the whole test split the transformer and a "
                                      f"TF-IDF logistic regression are statistically indistinguishable "
                                      f"(p = {F['sig_lr']['all']['p_value']:.2f}) — one of the "
                                      f"project's findings."),
                variant="flat"), unsafe_allow_html=True)

        # ---- live scoreboard
        st.markdown('<hr class="t-rule">', unsafe_allow_html=True)
        st.markdown(h2("Not convinced by one question? Score a batch.",
                       "Runs the model over a random sample of held-out test questions it has "
                       "never seen, and compares every answer with the reference."),
                    unsafe_allow_html=True)
        s1, s2, s3 = st.columns([1, 1, 1.4], gap="medium")
        with s1:
            n_eval = st.selectbox("Sample size", [25, 50, 100, 200], index=2)
        with s2:
            tier_eval = st.selectbox("Label tier", ["all", "constitution", "statute"], index=0)
        with s3:
            st.markdown('<div style="height:25px"></div>', unsafe_allow_html=True)
            go = st.button("Score the sample", width="stretch")

        if go or "ev_key" in st.session_state:
            key = (n_eval, tier_eval)
            if go:
                st.session_state["ev_key"] = key
            n_e, t_e = st.session_state.get("ev_key", key)
            with st.spinner("Running one batched pass over the sample …"):
                ev = engine.quick_eval(n=n_e, tier=t_e, seed=0)
            acc, top5 = float(ev.correct.mean()), float(ev.in_top5.mean())
            st.markdown(metrics([
                (f"{acc * 100:.0f}%", "", f"Exact match on {len(ev)} unseen questions", True),
                (f"{top5 * 100:.0f}%", "", "Reference provision inside the top 5", False),
                (f"{int(ev.correct.sum())}", f"/{len(ev)}", "Questions answered exactly right", False),
                (f"{ev.ms.iloc[0]:.1f}", "ms", "Per question, batched single pass", False),
            ]), unsafe_allow_html=True)
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            show = ev.assign(
                Result=ev.correct.map({True: "correct", False: "miss"}),
                Question=ev.question.str.slice(0, 110) + "…",
                Confidence=(ev.confidence * 100).round(0).astype(int).astype(str) + "%",
            )[["Result", "Question", "pred_name", "gold_name", "Confidence", "tier"]].rename(
                columns={"pred_name": "Model answer", "gold_name": "Reference answer", "tier": "Tier"})
            st.dataframe(show, width="stretch", hide_index=True, height=330)
            st.caption(f"A single deterministic pass, the same setting that scores "
                       f"{F['acc'] * 100:.1f}% over the full 722-question test split.")


# =========================================================================== RESULT
with tab_result:
    st.markdown(h2("The finding, in one line",
                   "Three questions were asked of this system. Only one of them has a "
                   "genuinely positive answer."), unsafe_allow_html=True)

    tt = F["tt"]
    ad_calls = float(tt.loc["adaptive_confidence", "avg_calls"])
    ad_acc = float(tt.loc["adaptive_confidence", "accuracy_mean"])
    f16_acc = float(tt.loc["fixed_16", "accuracy_mean"])
    one_acc = float(tt.loc["single", "accuracy_mean"])
    rnd_acc = float(tt.loc["ablation_random_routing", "accuracy_mean"])

    st.markdown(card("", f"""
<p style="font-family:var(--serif);font-size:24px;line-height:1.4;margin:0;letter-spacing:-.01em">
Spending more compute per question <b>did not</b> make this model more accurate — but
<b>choosing where to spend it</b> delivered the accuracy of a 16-pass budget for
<b>{ad_calls:.1f} passes</b>, a <b>{SAVING * 100:.0f}% saving</b>.</p>
<p class="t-sub" style="margin-top:16px">Accuracy at 1, 4, 8 and 16 passes sits inside the
spread across random seeds, so the extra sampling buys nothing measurable. The confidence
router is still useful: it keeps full accuracy at under half the cost, and it beats random
routing with identical bucket shares ({ad_acc * 100:.2f}% vs {rnd_acc * 100:.2f}%). The honest
read is that accuracy here is limited by the label space, not by the sampling budget.</p>"""),
                unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    order = [("single", "1 pass"), ("fixed_4", "4 passes"), ("fixed_8", "8 passes"),
             ("fixed_16", "16 passes"), ("adaptive_confidence", "Adaptive router"),
             ("ablation_random_routing", "Ablation: random routing"),
             ("ablation_length_routing", "Ablation: route by question length")]
    peak = max(float(tt.loc[k, "accuracy_mean"]) for k, _ in order if k in tt.index)
    rows = ""
    for key, name in order:
        if key not in tt.index:
            continue
        r = tt.loc[key]
        acc, calls = float(r.accuracy_mean), float(r.avg_calls)
        sd = float(r.accuracy_std) if r.accuracy_std == r.accuracy_std else 0.0
        hl = key == "adaptive_confidence"
        soft = " soft" if key.startswith("ablation") else ""
        rows += (f'<tr class="{"hl" if hl else ""}"><td class="nm">{esc(name)}</td>'
                 f'<td class="num">{calls:.1f}</td>'
                 f'<td class="num">{float(r.ms_per_question):.1f} ms</td>'
                 f'<td style="width:28%"><div class="t-bar{soft}">'
                 f'<i style="width:{acc / peak * 100:.1f}%"></i></div></td>'
                 f'<td class="num strong">{acc * 100:.2f}% ± {sd * 100:.2f}</td></tr>')
    st.markdown(card("Compute budget vs accuracy · test split, averaged over 3 sampling seeds",
                     f'<table class="t-tbl"><thead><tr><th>Strategy</th><th>Passes</th>'
                     f'<th>Latency</th><th>Accuracy</th><th></th></tr></thead>'
                     f'<tbody>{rows}</tbody></table>'
                     f'<p class="t-sub" style="margin-top:14px">McNemar test, adaptive vs 16 passes: '
                     f'p = {F["sig_tt"]["adaptive_vs_fixed16"]["p_value"]:.3f} on '
                     f'{F["sig_tt"]["adaptive_vs_fixed16"]["n"]} questions — no reliable accuracy '
                     f'difference, which is exactly the point: the saving is free.</p>'),
                unsafe_allow_html=True)

    st.markdown('<hr class="t-rule">', unsafe_allow_html=True)
    st.markdown(h2("How good is 30% really?",
                   "On a 1,932-way choice with a long tail of rare provisions, this is the "
                   "number that matters most for reading the demo."), unsafe_allow_html=True)

    g1, g2 = st.columns([1.3, 1], gap="medium")
    with g1:
        comp = [("Random guess", CHANCE, False), ("TF-IDF + nearest neighbours", F["knn"], False),
                ("TF-IDF + logistic regression", F["lr"], False),
                ("InLegalBERT, fine-tuned", F["acc"], True)]
        top = max(c[1] for c in comp)
        crows = ""
        for name, v, hl in comp:
            crows += (f'<tr class="{"hl" if hl else ""}"><td class="nm">{esc(name)}</td>'
                      f'<td style="width:42%"><div class="t-bar{"" if hl else " soft"}">'
                      f'<i style="width:{max(v / top * 100, 0.4):.1f}%"></i></div></td>'
                      f'<td class="num strong">{v * 100:.2f}%</td></tr>')
        st.markdown(card("Accuracy ladder · 722 held-out test questions",
                         f'<table class="t-tbl"><thead><tr><th>System</th><th>Exact match</th>'
                         f'<th></th></tr></thead><tbody>{crows}</tbody></table>'
                         f'<p class="t-sub" style="margin-top:14px">The transformer is '
                         f'{F["acc"] / CHANCE:.0f}× better than chance, but only '
                         f'{(F["acc"] - F["lr"]) * 100:.1f} points above a TF-IDF logistic '
                         f'regression — and that gap is not statistically significant. Pretrained '
                         f'legal-domain weights did not pay off here.</p>'), unsafe_allow_html=True)
    with g2:
        st.markdown(card("Where the difficulty really lives", f"""
<table class="t-tbl"><thead><tr><th>Label tier</th><th>Exact</th><th>Top-5</th></tr></thead><tbody>
<tr class="hl"><td class="nm">Constitution Articles</td><td class="num strong">{F['con'] * 100:.1f}%</td>
  <td class="num">{F['con_top5'] * 100:.1f}%</td></tr>
<tr><td class="nm">Statute Sections</td><td class="num strong">{F['sta'] * 100:.1f}%</td>
  <td class="num">{F['sta_top5'] * 100:.1f}%</td></tr>
</tbody></table>
<p class="t-sub" style="margin-top:14px">Articles are a small, well-covered set. Statute Sections
span hundreds of Acts, and <b>136 of 378</b> Section labels in the test split never appear in
training at all, so no classifier of this design can get them right. That, not the compute
budget, is the ceiling.</p>"""), unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown(card("What would move the number, if this were taken further", """
<div class="t-kv">
  <div><span class="k">Retrieval instead of classification</span>
       <span class="v">handles unseen Sections</span></div>
  <div><span class="k">Generative model that cites, then verifies</span>
       <span class="v">no fixed label space</span></div>
  <div><span class="k">Hierarchical labels: pick the Act, then the Section</span>
       <span class="v">splits a 1,932-way choice in two</span></div>
  <div><span class="k">Router on a stronger signal than dropout variance</span>
       <span class="v">dropout is a weak difficulty proxy</span></div>
</div>"""), unsafe_allow_html=True)


# =========================================================================== HOW
with tab_how:
    st.markdown(h2("From 47,789 raw answers to a scored task",
                   "The dataset ships as free text with no labels and no splits. Most of the work "
                   "was turning it into something that can be scored objectively."),
                unsafe_allow_html=True)

    w1, w2 = st.columns([1, 1], gap="medium")
    with w1:
        st.markdown(card("Building the task", steps([
            ("Start from question / answer pairs",
             "47,789 records of an Indian-law question with a free-text reference answer."),
            ("Read the label off the answer, not the question",
             "The gold label is the <b>first provision cited in the reference answer</b>, extracted "
             "by one regex-based rule that is later applied to model output too, so scoring is "
             "symmetric."),
            ("Throw away questions that give the answer away",
             "If the question already names the provision (including forms like “IPC 354”), the "
             "model could copy it, so the question is dropped from Task A."),
            ("Split so paraphrases cannot leak",
             "Near-duplicate questions (TF-IDF cosine ≥ 0.80) are grouped and a group never spans "
             "splits. Highest test→train similarity after splitting: <b>0.7998</b>."),
            ("Fine-tune the encoder",
             f"law-ai/InLegalBERT + dropout + a linear head over <b>{N_CLASSES:,}</b> provisions, "
             f"early-stopped on validation accuracy."),
            ("Route the compute",
             "A deterministic pass gives a confidence; validation quantiles turn it into an "
             "easy / medium / hard bucket worth 2 / 8 / 16 Monte-Carlo-dropout passes, which are "
             "averaged."),
        ])), unsafe_allow_html=True)

    with w2:
        st.markdown(card("The data behind the demo", kv([
            ("Training questions", f"{len(FRAMES['train']):,}"),
            ("Validation questions", f"{len(FRAMES['val']):,}"),
            ("Held-out test questions", f"{len(FRAMES['test']):,}"),
            ("Distinct provisions (classes)", f"{N_CLASSES:,}"),
            ("Chance accuracy", f"{CHANCE * 100:.2f}%"),
        ])), unsafe_allow_html=True)
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        st.markdown(card("The model being served right now", kv([
            ("Encoder", "law-ai/InLegalBERT"),
            ("Pooling / dropout", f"{RC.pooling} · {RC.dropout:g}"),
            ("Encoder LR / head LR", f"{RC.lr:g} · {RC.head_lr:g}"),
            ("Max question length", f"{RC.max_length} tokens"),
            ("Device", WARM["device"].upper()),
            ("Checkpoint load time", f"{WARM['load_seconds']:.1f} s"),
        ])), unsafe_allow_html=True)
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        st.markdown(banner("info", "·",
                           "Live predictions run in fp32 and were checked against the recorded "
                           "evaluation: <b>80 / 80</b> identical predictions. What you see in the "
                           "demo is the same model that produced the reported numbers."),
                    unsafe_allow_html=True)

    st.markdown('<hr class="t-rule">', unsafe_allow_html=True)
    st.markdown(h2("Reading a label", "Labels are stored in a compact key form."),
                unsafe_allow_html=True)
    st.markdown(card("", """
<div class="t-kv">
  <div><span class="k"><code>art:21</code></span><span class="v">Article 21, Constitution of India</span></div>
  <div><span class="k"><code>sec:138|negotiable instruments act</code></span>
       <span class="v">Section 138, Negotiable Instruments Act</span></div>
  <div><span class="k">Scoring rule</span>
       <span class="v">exact match on the first-cited provision</span></div>
</div>""", variant="flat"), unsafe_allow_html=True)

    plots = sorted((ROOT / "plots" / "eda").glob("*.png"))
    if plots:
        with st.expander("Exploratory plots from the data audit"):
            cols = st.columns(3, gap="medium")
            for i, p in enumerate(plots):
                with cols[i % 3]:
                    st.image(str(p), caption=p.stem.replace("_", " "), width="stretch")

    st.markdown('<hr class="t-rule">', unsafe_allow_html=True)
    st.markdown('<p class="t-sub">Research prototype for coursework. Not legal advice: the model '
                'is wrong about two questions in three, and it cannot cite a provision that was '
                'absent from its training labels.</p>', unsafe_allow_html=True)
