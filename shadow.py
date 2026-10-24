"""Shadow-mode router: watches kubernetes/kubernetes, predicts a SIG for every new issue,
records what it WOULD have done, and never touches the real repo.

Deploy (from your own Terminal, in the project folder):
    python3 -m modal secret create k8s-router --from-dotenv .env --force
    python3 -m modal deploy shadow.py
Check:
    python3 -m modal run shadow.py::report
    python3 -m modal volume get k8s-router /shadow data/ --force     # local copy of the database

Pipeline per new issue (first sighting only; issue id is the idempotency key):
    clean text -> M2 ModernBERT (CPU) -> Haiku -> cascade decision -> SQLite row
A daily job back-fills the human outcome (which sig/* labels appeared, when, and who chose them).
"""
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import modal

app = modal.App("k8s-router-shadow")
vol = modal.Volume.from_name("k8s-router", create_if_missing=True)
secret = modal.Secret.from_name("k8s-router")  # GITHUB_TOKEN, ANTHROPIC_API_KEY
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "transformers==4.48.3", "numpy<2.2", "anthropic>=0.70",
                      "requests>=2.31", "pyyaml>=6")
         .pip_install("scikit-learn==1.6.1")
         .add_local_file("data/raw/sigs.yaml", "/root/data/raw/sigs.yaml")
         .add_local_file("data/raw/owners_map.json", "/root/data/raw/owners_map.json")
         .add_local_file("data/processed/dataset.jsonl", "/root/data/processed/dataset.jsonl")
         .add_local_python_source("router", "fetch_issues", "agent"))
MNT = "/vol"
DB = f"{MNT}/shadow/shadow.db"

# ---- cascade config: thresholds chosen on val (see RESULTS.md); bump the version when they change
CONFIG = {
    "version": "v4-2026-10-24",
    "m2_run": "modernbert-s0",
    "m2_temperature": 1.171,   # fitted on val for this run (m2_eval.py modernbert-s0)
    "t_m2_alone": 0.99,        # M2 confident enough to route by itself
    "t_m2_agree": 0.69,        # below that, route only if Haiku picks the same SIG and M2 >= this
    "llm_cache": False,        # live traffic is ~1 issue/hour, so a 5-min cache is never hit and writes cost 1.25x
    "llm_model": "claude-haiku-5-5",
    "daily_llm_budget_usd": 1.00,  # over budget -> M2-only mode for the rest of the UTC day
    "min_body_words": 30,      # thinner issues go straight to a human (4% of issues; models are 60% right there)
    "agent_model": "claude-haiku-5-5",
    "t_agent": 0.71,           # agent answer trusted at/above this (picked on val, see agent_eval.py score)
}

ISSUES_Q = """
query($cursor: String) {
  repository(owner: "kubernetes", name: "kubernetes") {
    issues(first: 50, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { id number title body createdAt author { login } labels(first: 30) { nodes { name } } }
    }
  }
}"""

OUTCOME_Q = """
query($ids: [ID!]!) {
  nodes(ids: $ids) { ... on Issue {
    number state closedAt
    timelineItems(first: 100, itemTypes: [LABELED_EVENT, UNLABELED_EVENT, ISSUE_COMMENT]) {
      nodes { __typename
        ... on LabeledEvent { createdAt label { name } actor { login } }
        ... on UnlabeledEvent { createdAt label { name } }
        ... on IssueComment { createdAt body author { login } }
      } } } } }"""


def db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS predictions (
      issue_id TEXT PRIMARY KEY, number INTEGER, created_at TEXT, seen_at TEXT, text_hash TEXT,
      title TEXT, author TEXT, labels_at_seen TEXT,
      m2_sig TEXT, m2_conf REAL, m2_ms REAL,
      llm_sig TEXT, llm_conf REAL, llm_reason TEXT, llm_ms REAL, llm_in_tok INTEGER,
      llm_out_tok INTEGER, llm_cache_read INTEGER, llm_cost REAL, llm_error TEXT,
      decision TEXT, decided_sig TEXT, total_ms REAL, config_version TEXT);
    CREATE TABLE IF NOT EXISTS outcomes (
      number INTEGER PRIMARY KEY, checked_at TEXT, state TEXT, sig_labels TEXT,
      first_sig_at TEXT, routed_by TEXT);
    CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
    """)
    have = {r[1] for r in con.execute("PRAGMA table_info(predictions)")}
    for col, typ in (("agent_sig", "TEXT"), ("agent_conf", "REAL"), ("agent_reason", "TEXT"), ("agent_tools", "TEXT"),
                     ("agent_ms", "REAL"), ("agent_cost", "REAL"), ("agent_error", "TEXT")):
        if col not in have:  # additive migration; old rows keep NULLs
            con.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
    return con


def kv_get(con, k, default=None):
    r = con.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def kv_set(con, k, v):
    con.execute("INSERT INTO kv VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))


def body_words(text):
    """English words in the body, not counting the title or the issue template's ### headings."""
    import re
    body = text.split("\n", 1)[1] if "\n" in text else ""
    body = re.sub(r"###[^\n]*", "", body)
    return len(re.findall(r"[A-Za-z]{2,}", body))


def too_short(text):
    return body_words(text) < CONFIG["min_body_words"]


def decide(m2_sig, m2_conf, llm, text=None):
    if text is not None and too_short(text):
        return "human", None
    if m2_conf >= CONFIG["t_m2_alone"]:
        return "auto", m2_sig
    if llm and llm.get("sig") == m2_sig and m2_conf >= CONFIG["t_m2_agree"]:
        return "auto", m2_sig
    return "human", None


class M2:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = f"{MNT}/runs/{CONFIG['m2_run']}"
        self.classes = json.load(open(f"{path}/meta.json"))["classes"]
        self.tok = AutoTokenizer.from_pretrained(f"{path}/model")
        self.model = AutoModelForSequenceClassification.from_pretrained(f"{path}/model").eval()
        self.torch = torch

    def __call__(self, text):
        t = time.time()
        with self.torch.no_grad():
            x = self.tok(text, truncation=True, max_length=512, return_tensors="pt")
            logits = self.model(**x).logits[0] / CONFIG["m2_temperature"]
            p = self.torch.softmax(logits, -1)
        i = int(p.argmax())
        self._last = p
        return self.classes[i], float(p[i]), (time.time() - t) * 1000

    def top3(self, text):
        p = self._last  # probabilities from the __call__ on this same text
        idx = p.argsort(descending=True)[:3].tolist()
        return [(self.classes[i], float(p[i])) for i in idx]


@app.function(image=image, volumes={MNT: vol}, secrets=[secret], cpu=2, memory=4096, timeout=900,
              schedule=modal.Period(minutes=10), max_containers=1)
def poll():
    import anthropic
    os.chdir("/root")  # router.tools reads data/... relative paths
    from agent import InProcess, run as run_agent
    from fetch_issues import GraphQL
    from router.data import clean_text
    from router.llm import classify, sig_catalog

    con = db()
    now = datetime.now(timezone.utc)
    start = kv_get(con, "start_at")
    if not start:  # first run: only issues created from now on count as "live"
        start = (now - timedelta(minutes=30)).isoformat()
        kv_set(con, "start_at", start)
        con.commit()
    gql = GraphQL(os.environ["GITHUB_TOKEN"])
    seen = {r[0] for r in con.execute("SELECT issue_id FROM predictions")}
    new, cursor = [], None
    while True:  # newest first; stop at the first issue older than start
        d = gql.run(ISSUES_Q, {"cursor": cursor})["repository"]["issues"]
        stop = False
        for n in d["nodes"]:
            if n["createdAt"] < start:
                stop = True
                break
            if n["id"] not in seen:
                new.append(n)
        if stop or not d["pageInfo"]["hasNextPage"]:
            break
        cursor = d["pageInfo"]["endCursor"]
    print(f"{len(new)} new issues since last run")
    if not new:
        return

    m2 = M2()
    labels, catalog = sig_catalog("/root/data/raw/sigs.yaml")
    client = anthropic.Anthropic(max_retries=4)   # SDK retries 429/5xx with exponential backoff
    day = now.strftime("%Y-%m-%d")
    for n in reversed(new):  # oldest first
        t0 = time.time()
        text = clean_text(n["title"], n["body"] or "")
        m2_sig, m2_conf, m2_ms = m2(text)
        spent = float(kv_get(con, f"llm_cost:{day}", 0))
        llm, err = None, None
        if spent < CONFIG["daily_llm_budget_usd"]:
            try:
                llm = classify(client, CONFIG["llm_model"], labels, catalog, text, cache=CONFIG["llm_cache"])
                kv_set(con, f"llm_cost:{day}", spent + llm["cost"])
            except Exception as e:  # LLM down -> cascade falls back to M2-only for this issue
                err = f"{type(e).__name__}: {str(e)[:200]}"
        else:
            err = "budget"
        decision, sig = decide(m2_sig, m2_conf, llm, text)
        ag, ag_err = {}, None
        if decision == "human" and too_short(text):
            ag_err = "too_short"   # no point asking the agent about an empty issue
        elif decision == "human":  # third layer: only for issues the cheap layers would hand to a person
            spent = float(kv_get(con, f"llm_cost:{day}", 0))
            if spent < CONFIG["daily_llm_budget_usd"]:
                try:
                    ag = run_agent(client, CONFIG["agent_model"], labels, text, n["createdAt"],
                                   m2.top3(text), (llm or {}).get("sig"), InProcess())
                    kv_set(con, f"llm_cost:{day}", spent + ag["cost"])
                    if ag["sig"] != "none" and not ag.get("duplicate_of") and ag["confidence"] >= CONFIG["t_agent"]:
                        decision, sig = "auto", ag["sig"]
                except Exception as e:
                    ag_err = f"{type(e).__name__}: {str(e)[:200]}"
            else:
                ag_err = "budget"
        llm = llm or {}
        con.execute("INSERT OR IGNORE INTO predictions (issue_id, number, created_at, seen_at, text_hash, title, "
                    "author, labels_at_seen, m2_sig, m2_conf, m2_ms, llm_sig, llm_conf, llm_reason, llm_ms, llm_in_tok, "
                    "llm_out_tok, llm_cache_read, llm_cost, llm_error, decision, decided_sig, total_ms, config_version) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            n["id"], n["number"], n["createdAt"], datetime.now(timezone.utc).isoformat(),
            hashlib.sha256(text.encode()).hexdigest()[:16], n["title"], (n.get("author") or {}).get("login"),
            json.dumps([l["name"] for l in n["labels"]["nodes"]]),
            m2_sig, m2_conf, m2_ms, llm.get("sig"), llm.get("confidence"), llm.get("reason"), llm.get("ms"),
            llm.get("in_tok"), llm.get("out_tok"), llm.get("cache_read"), llm.get("cost"), err,
            decision, sig, (time.time() - t0) * 1000, CONFIG["version"]))
        con.execute("UPDATE predictions SET agent_sig=?, agent_conf=?, agent_reason=?, agent_tools=?, agent_ms=?, "
                    "agent_cost=?, agent_error=? WHERE issue_id=?",
                    (ag.get("sig"), ag.get("confidence"), ag.get("reason"), json.dumps(ag.get("tools", [])),
                     ag.get("ms"), ag.get("cost"), ag_err, n["id"]))
        con.commit()  # one row at a time: a crash mid-batch never loses or duplicates work
        print(f"#{n['number']}: M2 {m2_sig} {m2_conf:.2f} | LLM {llm.get('sig')} | {decision} {sig or ''}")
    con.close()
    vol.commit()


@app.function(image=image, volumes={MNT: vol}, secrets=[secret], timeout=900,
              schedule=modal.Cron("17 * * * *"), max_containers=1)
def backfill():
    """Hourly: record which SIG humans actually chose, how fast, and who chose it."""
    import re
    from fetch_issues import GraphQL
    sig_cmd = re.compile(r"^[ \t>]*/sig\s+([\w-]+)", re.M | re.I)
    con = db()
    rows = con.execute("SELECT issue_id, number, created_at, author FROM predictions WHERE created_at > ?",
                       ((datetime.now(timezone.utc) - timedelta(days=35)).isoformat(),)).fetchall()
    gql = GraphQL(os.environ["GITHUB_TOKEN"])
    meta = {r[0]: r for r in rows}
    ids = list(meta)
    for k in range(0, len(ids), 40):
        for node in gql.run(OUTCOME_Q, {"ids": ids[k:k + 40]})["nodes"]:
            if not node:
                continue
            n = node["number"]
            author = next(r[3] for r in rows if r[1] == n)
            cur, first_at, routed_by = set(), None, None
            for e in sorted(node["timelineItems"]["nodes"], key=lambda e: e["createdAt"]):
                t = e["__typename"]
                lab = ((e.get("label") or {}).get("name") or "")
                if t == "LabeledEvent" and lab.startswith("sig/"):
                    cur.add(lab[4:])
                    first_at = first_at or e["createdAt"]
                elif t == "UnlabeledEvent" and lab.startswith("sig/"):
                    cur.discard(lab[4:])
                elif t == "IssueComment" and routed_by is None and sig_cmd.search(e["body"] or ""):
                    routed_by = "author_comment" if (e.get("author") or {}).get("login") == author else "other"
            con.execute("INSERT INTO outcomes VALUES (?,?,?,?,?,?) ON CONFLICT(number) DO UPDATE SET "
                        "checked_at=excluded.checked_at, state=excluded.state, sig_labels=excluded.sig_labels, "
                        "first_sig_at=excluded.first_sig_at, routed_by=excluded.routed_by",
                        (n, datetime.now(timezone.utc).isoformat(), node["state"], json.dumps(sorted(cur)),
                         first_at, routed_by))
    con.commit()
    con.close()
    vol.commit()
    print(f"outcomes updated for {len(ids)} issues")


@app.function(image=image, volumes={MNT: vol}, timeout=300)
def report():
    """Live numbers so far. 'labeled' = humans have put at least one sig label on the issue."""
    vol.reload()
    con = db()
    q = con.execute("""SELECT p.number, p.decision, p.decided_sig, p.m2_sig, p.llm_sig, p.total_ms,
                              p.llm_cost, p.created_at, o.sig_labels, o.first_sig_at, o.routed_by
                       FROM predictions p LEFT JOIN outcomes o ON o.number = p.number""").fetchall()
    n = len(q)
    lab = [r for r in q if r[8] and json.loads(r[8])]
    auto = [r for r in lab if r[1] == "auto"]
    hits = sum(r[2] in json.loads(r[8]) for r in auto)
    cost = sum(r[6] or 0 for r in q)
    lat = [r[5] for r in q if r[5]]
    print(f"start {kv_get(con, 'start_at')}  issues seen {n}  with human sig label {len(lab)}")
    if lab:
        print(f"auto-routed {len(auto)}/{len(lab)} = {len(auto)/len(lab):.1%}, precision "
              f"{hits}/{len(auto)} = {hits/max(len(auto),1):.1%}")
        print(f"M2 top-1 {sum(r[3] in json.loads(r[8]) for r in lab)/len(lab):.1%}, "
              f"LLM top-1 {sum((r[4] or '') in json.loads(r[8]) for r in lab)/len(lab):.1%}")
    if lat:
        import statistics
        p50 = statistics.median(lat) / 1000
        p95 = sorted(lat)[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))] / 1000  # nearest-rank
        print(f"latency p50 {p50:.1f}s p95 {p95:.1f}s (n={len(lat)}); "
              f"LLM cost ${cost:.4f} total, ${1000*cost/max(n,1):.2f} per 1k issues")
    ag = con.execute("SELECT COUNT(agent_sig), SUM(agent_cost), COUNT(agent_error) FROM predictions").fetchone()
    print(f"agent runs {ag[0]}, agent cost ${ag[1] or 0:.4f}, agent errors {ag[2]}")
    by_ver = con.execute("SELECT config_version, COUNT(*) FROM predictions GROUP BY 1").fetchall()
    print("rows by config version:", dict(by_ver))
