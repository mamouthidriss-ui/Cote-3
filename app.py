"""
Cotes du week-end : favoris à cote 1,30-1,50 dans les 5 grands championnats.
Les 3 meilleurs (probabilité du modèle) par samedi et par dimanche.
Données gratuites : football-data.co.uk (résultats + fixtures.csv avec cotes).
"""
import io
import threading
import time
import datetime as dt
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string
from sklearn.linear_model import LogisticRegression

LEAGUES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A",
           "D1": "Bundesliga", "F1": "Ligue 1"}
ODDS_MIN, ODDS_MAX, TOP_N = 1.30, 1.50, 3
BASE = "https://www.football-data.co.uk"
TTL = 6 * 3600            # rafraîchissement des données toutes les 6 h
WARMUP = 500              # matchs ignorés à l'entraînement (Elo pas encore stable)
HIST_COLS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HST", "AST"]
FEATS = ["elo", "pts", "gd", "sot", "rest"]

app = Flask(__name__)
_cache = {"data": None, "ts": 0.0, "error": None}
_lock = threading.Lock()


# ------------------------------------------------------------------ données
def fetch_csv(url):
    r = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    try:
        txt = r.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        txt = r.content.decode("latin-1")
    df = pd.read_csv(io.StringIO(txt), on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    return df.dropna(how="all")


def season_codes(today):
    y = today.year % 100
    cur = (y, y + 1) if today.month >= 7 else (y - 1, y)
    return [f"{a:02d}{b:02d}" for a, b in ((cur[0] - 1, cur[1] - 1), cur)]


def parse_dates(s):
    return pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed")


def load_history(today):
    frames = []
    for season in season_codes(today):
        for code in LEAGUES:
            try:
                df = fetch_csv(f"{BASE}/mmz4281/{season}/{code}.csv")
            except Exception:
                continue
            frames.append(clean_history(df, code))
    if not frames:
        raise RuntimeError("Impossible de télécharger l'historique.")
    return pd.concat(frames).sort_values("Date").reset_index(drop=True)


def clean_history(df, code=None):
    df = df.copy()
    for c in HIST_COLS:
        if c not in df:
            df[c] = np.nan
    if code:
        df["Div"] = code
    df["Date"] = parse_dates(df["Date"])
    df = df.dropna(subset=["Date", "FTR", "FTHG", "FTAG", "HomeTeam", "AwayTeam"])
    return df[HIST_COLS]


def load_fixtures():
    fx = fetch_csv(f"{BASE}/fixtures.csv")
    return clean_fixtures(fx)


def clean_fixtures(fx):
    fx = fx.copy()
    for c in ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365A", "MaxH", "MaxA"]:
        if c not in fx:
            fx[c] = np.nan
    fx["Date"] = parse_dates(fx["Date"])
    fx = fx[fx["Div"].isin(LEAGUES) & fx["Date"].notna()]
    return fx


# ------------------------------------------------------------------ modèle
class Model:
    """Elo (écart de buts) + forme 6 matchs + repos -> régression logistique."""

    def __init__(self, hist):
        self.elo = defaultdict(lambda: 1500.0)
        self.hist = defaultdict(lambda: deque(maxlen=6))
        self.last = {}
        X, y = [], []
        for r in hist.itertuples(index=False):
            h, a = (r.Div, r.HomeTeam), (r.Div, r.AwayTeam)
            f = self._feats(h, a, r.Date)
            if f is not None:
                X.append(f)
                y.append({"H": 0, "D": 1, "A": 2}[r.FTR])
            self._update(h, a, r)
        X, y = np.array(X), np.array(y)
        X, y = X[WARMUP:], y[WARMUP:]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.clf = LogisticRegression(C=0.5, max_iter=500).fit((X - self.mu) / self.sd, y)
        self.n_train = len(y)

    def _roll(self, t, j):
        q = self.hist[t]
        return np.mean([z[j] for z in q]) if len(q) >= 3 else 0.0

    def _rest(self, t, date):
        return min((date - self.last[t]).days, 10) if t in self.last else 7

    def _feats(self, h, a, date):
        if len(self.hist[h]) < 3 or len(self.hist[a]) < 3:
            return None
        return [self.elo[h] - self.elo[a] + 60,
                self._roll(h, 0) - self._roll(a, 0),
                self._roll(h, 1) - self._roll(a, 1),
                self._roll(h, 2) - self._roll(a, 2),
                self._rest(h, date) - self._rest(a, date)]

    def _update(self, h, a, r):
        gd = r.FTHG - r.FTAG
        s = {"H": 1, "D": .5, "A": 0}[r.FTR]
        ph = 1 / (1 + 10 ** (-(self.elo[h] - self.elo[a] + 60) / 400))
        k = 20 * (np.log(abs(gd) + 1) + 1)
        self.elo[h] += k * (s - ph)
        self.elo[a] -= k * (s - ph)
        pts_h = {"H": 3, "D": 1, "A": 0}[r.FTR]
        pts_a = {"A": 3, "D": 1, "H": 0}[r.FTR]
        sh = (r.HST - r.AST) if pd.notna(r.HST) and pd.notna(r.AST) else gd
        self.hist[h].append((pts_h, gd, sh))
        self.hist[a].append((pts_a, -gd, -sh))
        self.last[h] = self.last[a] = r.Date

    def predict(self, div, home, away, date):
        f = self._feats((div, home), (div, away), date)
        if f is None:
            return None
        return self.clf.predict_proba(((np.array([f]) - self.mu) / self.sd))[0]  # H, D, A


# ------------------------------------------------------------------ sélection
def build_picks(model, fixtures, today):
    rows = []
    for r in fixtures.itertuples(index=False):
        if r.Date.date() < today or r.Date.dayofweek not in (5, 6):
            continue
        p = model.predict(r.Div, r.HomeTeam, r.AwayTeam, r.Date)
        if p is None:
            continue
        for side, team, odds, best, prob in (
                ("H", r.HomeTeam, r.B365H, r.MaxH, p[0]),
                ("A", r.AwayTeam, r.B365A, r.MaxA, p[2])):
            if pd.isna(odds) or not (ODDS_MIN <= odds <= ODDS_MAX):
                continue
            rows.append({
                "date": r.Date.strftime("%Y-%m-%d"),
                "jour": "Samedi" if r.Date.dayofweek == 5 else "Dimanche",
                "heure": r.Time if isinstance(r.Time, str) else "",
                "ligue": LEAGUES[r.Div],
                "match": f"{r.HomeTeam} - {r.AwayTeam}",
                "pari": f"{team} gagne",
                "cote": round(float(odds), 2),
                "meilleure_cote": round(float(best), 2) if pd.notna(best) else None,
                "proba": round(float(prob) * 100, 1),
                "valeur": round((float(prob) * float(odds) - 1) * 100, 1),
            })
    rows.sort(key=lambda x: (x["date"], -x["proba"]))
    out, count = [], defaultdict(int)
    for x in rows:
        if count[x["date"]] < TOP_N:
            out.append(x)
            count[x["date"]] += 1
    return out


def refresh():
    today = dt.date.today()
    hist = load_history(today)
    model = Model(hist)
    picks = build_picks(model, load_fixtures(), today)
    return {"picks": picks,
            "updated": dt.datetime.now().strftime("%d/%m/%Y %H:%M"),
            "n_matches": len(hist), "n_train": model.n_train}


def get_data(force=False):
    with _lock:
        if force or _cache["data"] is None or time.time() - _cache["ts"] > TTL:
            try:
                _cache["data"] = refresh()
                _cache["ts"] = time.time()
                _cache["error"] = None
            except Exception as e:  # on garde les anciennes données si possible
                _cache["error"] = str(e)
                _cache["ts"] = time.time() - TTL + 600   # nouvel essai dans 10 min
        return _cache["data"], _cache["error"]


# ------------------------------------------------------------------ web
PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cotes du week-end</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--tx:#14181f;--mu:#667085;--ac:#0f766e;--bd:#e4e7ec;--wa:#fff4e5}
@media (prefers-color-scheme:dark){:root{--bg:#0f1318;--card:#171c23;--tx:#eef1f5;--mu:#98a2b3;--ac:#2dd4bf;--bd:#27303b;--wa:#2a2112}}
*{box-sizing:border-box}body{margin:0;font:16px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--tx);padding:16px;max-width:640px;margin:auto}
h1{font-size:1.3rem;margin:4px 0 2px}h2{font-size:1.05rem;margin:22px 0 8px}
.sub{color:var(--mu);font-size:.85rem}.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:12px 14px;margin:8px 0}
.row{display:flex;justify-content:space-between;align-items:center;gap:10px}.pick{font-weight:600}
.odds{font-size:1.4rem;font-weight:700;color:var(--ac)}.meta{color:var(--mu);font-size:.82rem;margin-top:4px}
.warn{background:var(--wa);border-radius:10px;padding:10px 12px;font-size:.82rem;margin:12px 0}
.empty{color:var(--mu);padding:18px 0}
</style></head><body>
<h1>Cotes du week-end</h1>
<div class="sub">Favoris à cote {{omin}}–{{omax}} · 5 grands championnats · top {{n}} par jour</div>
{% if error %}<div class="warn">Erreur de mise à jour : {{error}}</div>{% endif %}
{% if not data %}<div class="empty">Chargement des données… recharge la page dans 30 secondes.</div>
{% else %}
  {% set days = {} %}
  {% for p in data.picks %}{% if p.date not in days %}{% set _ = days.update({p.date: []}) %}{% endif %}{% set _ = days[p.date].append(p) %}{% endfor %}
  {% for d, items in days.items() %}
    <h2>{{items[0].jour}} {{d[8:10]}}/{{d[5:7]}}</h2>
    {% for p in items %}
    <div class="card"><div class="row"><div><div class="pick">{{p.pari}}</div>
      <div class="sub">{{p.match}}</div></div><div class="odds">{{"%.2f"|format(p.cote)}}</div></div>
      <div class="meta">{{p.ligue}}{% if p.heure %} · {{p.heure}}{% endif %} · probabilité modèle {{p.proba}} %
      · valeur {{"%+.1f"|format(p.valeur)}} %{% if p.meilleure_cote %} · meilleure cote marché {{"%.2f"|format(p.meilleure_cote)}}{% endif %}</div></div>
    {% endfor %}
  {% endfor %}
  {% if not data.picks %}<div class="empty">Aucun pari éligible pour l'instant. Les cotes des matchs du week-end
  sont en général publiées à partir du jeudi.</div>{% endif %}
  <div class="sub" style="margin-top:14px">Mis à jour le {{data.updated}} · modèle entraîné sur {{data.n_train}} matchs</div>
{% endif %}
<div class="warn"><b>À lire :</b> outil d'analyse, pas une garantie de gains. Le backtest 2025-26 donne un ROI de
+6,9 % avec une marge d'erreur de ±11 points : l'avantage n'est pas prouvé. Les « 75 % de réussite » n'empêchent pas
les séries perdantes. Ne mise que ce que tu peux perdre. Jeu interdit aux mineurs. Aide : joueurs-info-service.fr</div>
</body></html>"""


@app.route("/")
def index():
    data, error = get_data()
    return render_template_string(PAGE, data=data, error=error,
                                  omin=f"{ODDS_MIN:.2f}", omax=f"{ODDS_MAX:.2f}", n=TOP_N)


@app.route("/api/picks")
def api():
    data, error = get_data()
    return jsonify({"data": data, "error": error})


@app.route("/refresh")
def force_refresh():
    data, error = get_data(force=True)
    return jsonify({"ok": error is None, "error": error})


@app.route("/health")
def health():
    return "ok"


threading.Thread(target=get_data, daemon=True).start()   # préchauffage au démarrage

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
