import os, json, re, math, difflib, unicodedata, datetime as dt
import numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import Ridge

try:
    from IPython.display import HTML, display
except Exception:
    HTML = display = None

from zoneinfo import ZoneInfo

ZONA = "America/New_York"   # zona de referencia (la NFL fecha sus juegos en hora del Este)
HOY = dt.datetime.now(ZoneInfo(ZONA)).date()   # Colab/GitHub corren en UTC: sin esto el Monday Night "ya es mañana" y se descarta
ESPN_PASO = 7         # Fútbol: tamaño (días) de cada ventana al buscar próximos partidos en ESPN
JUEGOS_MIN_SEMANA = 2  # NFL: si a la semana actual le quedan <= N juegos, se muestra la siguiente semana
N_PROX = 10           # Fútbol: los N partidos más cercanos por competición (sin importar la fecha)
ESPN_DIAS = 60        # Fútbol: hasta cuántos días hacia adelante buscar en ESPN
DIAS_RES = 7          # Resultados: partidos terminados en los últimos N días
N_RACHA = 5           # Racha: últimos N partidos de cada equipo
MODELO_IA = "gemini-2.5-flash"
SALIDA = "index.html"
TS = pd.Timestamp
DIAG = []             # mensajes de diagnóstico (se imprimen y se muestran en el HTML si una pestaña sale vacía)


def diag(msg):
    DIAG.append(msg)
    print("·", msg)


def num(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except Exception:
        return None


def fila(l, m, d, sd=None, line=None, o="Over", u="Under", p=None, real=None, dec=2):
    """dec=0 -> yardas y similares: proyección y real redondeados a entero."""
    m = round(float(m), dec) if dec else int(round(float(m)))
    if dec == 0 and real is not None:
        real = int(round(real))
    return dict(l=l, m=m, d=d, sd=sd, line=line, o=o, u=u, p=p, real=real)


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def racha(df, t, ch, ca, sh, sa, ref, n=N_RACHA):
    """Últimos n resultados del equipo t antes de ref: 'G' ganó, 'E' empató, 'P' perdió (del más viejo al más reciente)."""
    d = df[((df[ch] == t) | (df[ca] == t)) & (df.Date < ref)].sort_values("Date").tail(n)
    out = []
    for r in d.itertuples():
        gh, ga = getattr(r, sh), getattr(r, sa)
        gf, gc = (gh, ga) if getattr(r, ch) == t else (ga, gh)
        out.append("G" if gf > gc else ("E" if gf == gc else "P"))
    return out


def ml_dec(x):
    """Moneyline americano -> cuota decimal (internamente todo se guarda en decimal)."""
    x = num(x)
    if x is None or x == 0:
        return None
    return round(1 + x / 100, 3) if x > 0 else round(1 + 100 / abs(x), 3)


# ───────────────────────── Secretos / IA ─────────────────────────
def secreto(nombre):
    """Variable de entorno (GitHub Actions) o secreto de Colab (panel 🔑)."""
    k = os.getenv(nombre)
    if not k:
        try:
            from google.colab import userdata
            k = userdata.get(nombre)
        except Exception:
            k = None
    return k


def ia(fx, dep):
    k = secreto("GOOGLE_API_KEY")
    if not k or not fx:
        return {}, "", []
    try:
        from google import genai
        from google.genai import types
        esq = ('{"resumen":"3 frases","partidos":[{"id":"..","notas":"1 frase de contexto/lesiones"}],'
               '"picks":[{"id":"..","mercado":"..","eleccion":"..","cuota":2.1,"confianza":"alta|media|baja","razon":".."}]}')
        prompt = (f"Hoy es {HOY}. Eres analista de apuestas deportivas de {dep}. Te paso el modelo estadístico propio "
                  "de los próximos partidos: proyección, probabilidades del resultado con su cuota (decimal), y para cada "
                  "mercado la media del modelo y la línea de referencia.\n"
                  f"{json.dumps(fx, ensure_ascii=False)}\n"
                  "Busca en la web lesiones, alineaciones probables, clima y noticias recientes de esos partidos y "
                  "combínalo con el modelo. Devuelve SOLO un objeto JSON (sin markdown) con esta forma: " + esq + ". "
                  "Reglas: máximo 6 picks; solo si hay valor real (prob. del modelo × cuota > 1, o media del modelo "
                  "claramente por encima/debajo de la línea) y las noticias no lo contradicen; si no hay valor, picks vacío. "
                  "'eleccion' debe ser concreta (ej. 'Over 2.5 goles', 'Bills -3.5', 'Gana Real Madrid'). "
                  "'razon' máximo 25 palabras citando modelo y noticia. No inventes cuotas: usa las dadas o null.")
        r = genai.Client(api_key=k).models.generate_content(
            model=MODELO_IA, contents=prompt,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]))
        o = json.loads(re.search(r"\{.*\}", r.text, re.S).group(0))
        return ({x["id"]: x for x in o.get("partidos", [])}, o.get("resumen", ""), o.get("picks", []) or [])
    except Exception as e:
        print("IA omitida:", e)
        return {}, "", []


# ───────────────────────── Logos ─────────────────────────
NFL_LOGO = {"WAS": "wsh", "LA": "lar", "LAR": "lar", "OAK": "lv", "SD": "lac", "STL": "lar"}


LOGOS = {}   # fútbol: nombre normalizado -> URL del escudo (se llena con cargar_logos / ESPN)


def logo(eq, sp):
    if sp == "NFL":
        return f"https://a.espncdn.com/i/teamlogos/nfl/500/{NFL_LOGO.get(eq, eq.lower())}.png"
    return LOGOS.get(_n(eq), "")


# ───────────────────────── Modelo de tasas ─────────────────────────
def tasas(df, vh, va, k=4, ref=None, hl=120):
    ref = ref if ref is not None else TS(HOY)
    df = df.assign(w=0.5 ** ((ref - df.Date).dt.days / hl))
    mh, ma = np.average(df[vh], weights=df.w), np.average(df[va], weights=df.w)
    att, dfn = {}, {}
    for t in set(df.h) | set(df.a):
        H, Av = df[df.h == t], df[df.a == t]
        sw = H.w.sum() + Av.w.sum() + k
        att[t] = ((H.w * H[vh] / mh).sum() + (Av.w * Av[va] / ma).sum() + k) / sw
        dfn[t] = ((H.w * H[va] / ma).sum() + (Av.w * Av[vh] / mh).sum() + k) / sw
    return att, dfn, mh, ma


def esperado(m, h, a):
    att, dfn, mh, ma = m
    return mh * att.get(h, 1) * dfn.get(a, 1), ma * att.get(a, 1) * dfn.get(h, 1)


# ───────────────────────── NFL ─────────────────────────
NFL_TEAM = {"LAR": "LA", "STL": "LA", "OAK": "LV", "LVR": "LV", "SD": "LAC", "SDG": "LAC", "WSH": "WAS"}
POS = {"QB": ("passing_yards", "attempts", 10),
       "RB": ("rushing_yards", "carries", 8),
       "WR": ("receiving_yards", "targets", 4)}
SD_DEF = {"QB": 70.0, "RB": 30.0, "WR": 30.0}
SD_MARGEN = 13.5

DIVS = {"AFC Este": "BUF MIA NE NYJ", "AFC Norte": "BAL CIN CLE PIT", "AFC Sur": "HOU IND JAX TEN",
        "AFC Oeste": "DEN KC LV LAC", "NFC Este": "DAL NYG PHI WAS", "NFC Norte": "CHI DET GB MIN",
        "NFC Sur": "ATL CAR NO TB", "NFC Oeste": "ARI LA SF SEA"}
DIV = {t: d for d, ts in DIVS.items() for t in ts.split()}


def es_div(r, h, a):
    v = getattr(r, "div_game", None)          # columna de nflverse; si no existe, se usa el diccionario
    if v is not None and v == v:
        return bool(int(v))
    return DIV.get(h) is not None and DIV.get(h) == DIV.get(a)


def stats_nfl(nfr, seasons, sch):
    ps = nfr.load_player_stats(seasons).to_pandas()
    if "team" not in ps.columns and "recent_team" in ps.columns:
        ps = ps.rename(columns={"recent_team": "team"})
    if "player_display_name" not in ps.columns:
        ps["player_display_name"] = ps["player_name"]
    for c in ("passing_yards", "rushing_yards", "receiving_yards", "attempts", "carries", "targets"):
        if c not in ps.columns:
            ps[c] = 0
        ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0)
    ps["team"] = ps["team"].replace(NFL_TEAM)

    base = sch[["season", "week", "gameday", "home_team", "away_team"]]
    lg = pd.concat([base.rename(columns={"home_team": "team", "away_team": "opp"}),
                    base.rename(columns={"away_team": "team", "home_team": "opp"})])
    lg = lg[["season", "week", "gameday", "team", "opp"]].rename(columns={"gameday": "Date"})
    ps = ps.drop(columns=[c for c in ("opponent_team", "Date", "opp") if c in ps.columns]) \
           .merge(lg, on=["season", "week", "team"], how="left")

    eg = ps.groupby(["season", "week", "team"], as_index=False)[["passing_yards", "rushing_yards"]].sum()
    eg_d = {(int(r.season), int(r.week), r.team): (float(r.passing_yards), float(r.rushing_yards))
            for r in eg.itertuples()}

    jug = sch.dropna(subset=["home_score", "away_score"])
    wide = jug[["season", "week", "gameday", "home_team", "away_team"]] \
        .rename(columns={"gameday": "Date", "home_team": "h", "away_team": "a"})
    wide = wide.merge(eg.rename(columns={"team": "h", "passing_yards": "air_h", "rushing_yards": "gr_h"}),
                      on=["season", "week", "h"], how="left")
    wide = wide.merge(eg.rename(columns={"team": "a", "passing_yards": "air_a", "rushing_yards": "gr_a"}),
                      on=["season", "week", "a"], how="left")
    wide = wide.dropna(subset=["air_h", "air_a", "gr_h", "gr_a"])
    return ps, wide, eg_d


def modelo_nfl(jug, wide, ps, ref, eq, ix):
    jc = jug[jug.gameday < ref]
    w = 0.5 ** ((ref - jc.gameday).dt.days.values / 150)

    def X(signo):
        M = np.zeros((len(jc), len(eq)))
        for r, (h, a) in enumerate(zip(jc.home_team, jc.away_team)):
            M[r, ix[h]] = 1
            M[r, ix[a]] = signo
        return M

    M = dict(
        mm=Ridge(alpha=3).fit(X(-1), (jc.home_score - jc.away_score).values, sample_weight=w),
        mt=Ridge(alpha=3).fit(X(1), (jc.home_score + jc.away_score).values, sample_weight=w),
        ix=ix, air=None)

    if wide is not None and len(wide):
        wc = wide[wide.Date < ref]
        if len(wc) > 20:
            M["air"] = tasas(wc, "air_h", "air_a", k=3, ref=ref, hl=150)
            M["gr"] = tasas(wc, "gr_h", "gr_a", k=3, ref=ref, hl=150)
            M["sd_air"] = float(pd.concat([wc.air_h, wc.air_a]).std())
            M["sd_gr"] = float(pd.concat([wc.gr_h, wc.gr_a]).std())
            pc = ps[ps.Date < ref]
            M["ps"] = pc
            sdp = {}
            for pos, (y, v, thr) in POS.items():
                s = pc.loc[(pc.position == pos) & (pc[v] >= thr), y].std()
                sdp[pos] = float(s) if s == s else SD_DEF[pos]
            M["sd_pos"] = sdp
    return M


def pred_nfl(M, h, a):
    ix, mm, mt = M["ix"], M["mm"], M["mt"]
    mar = mm.intercept_ + mm.coef_[ix[h]] - mm.coef_[ix[a]]
    tot = mt.intercept_ + mt.coef_[ix[h]] + mt.coef_[ix[a]]
    return mar, tot


def jugador(M, team, pos, ref, dfn_opp):
    ycol, vcol, _ = POS[pos]
    d = M["ps"]
    d = d[d.position == pos]
    rec = d[(d.team == team) & (d.Date >= ref - dt.timedelta(days=42))]
    if rec.empty:
        rec = d[d.team == team]
    if rec.empty:
        return None
    g = rec.groupby(["player_id", "player_display_name"])[vcol].sum()
    if g.empty or g.max() <= 0:
        return None
    pid, nombre = g.idxmax()
    h = d[d.player_id == pid].sort_values("Date", ascending=False).head(8)
    n = len(h)
    w = 0.7 ** np.arange(n)
    mu = float(np.average(h[ycol], weights=w)) * dfn_opp
    var_j = float(h[ycol].var(ddof=0)) if n > 1 else 0.0
    sd = math.sqrt((n * var_j + 4 * M["sd_pos"][pos] ** 2) / (n + 4))
    return pid, nombre, mu, max(sd, 10.0)


def filas_nfl(h, a, mar, tot, sp_line, tot_line, M, ref, R=None):
    R = R or {}
    Y = lambda *a_, **k: fila(*a_, dec=0, **k)       # yardas: sin decimales
    rows = [
        fila("Spread · local gana por >", mar, "n", SD_MARGEN, num(sp_line), f"{h} cubre", f"{a} cubre", real=R.get("spread")),
        fila("Total de puntos", tot, "n", 13, num(tot_line), real=R.get("total")),
    ]
    if M.get("air"):
        eah, eaa = esperado(M["air"], h, a)
        egh, ega = esperado(M["gr"], h, a)
        rows += [
            Y(f"Yds aire {h}", eah, "n", M["sd_air"], real=R.get("air_h")),
            Y(f"Yds aire {a}", eaa, "n", M["sd_air"], real=R.get("air_a")),
            Y(f"Yds suelo {h}", egh, "n", M["sd_gr"], real=R.get("gr_h")),
            Y(f"Yds suelo {a}", ega, "n", M["sd_gr"], real=R.get("gr_a")),
        ]
        for team, opp in ((h, a), (a, h)):
            for pos, etq, key in (("QB", "yds pase", "air"), ("RB", "yds suelo", "gr"), ("WR", "yds recep.", "air")):
                j = jugador(M, team, pos, ref, M[key][1].get(opp, 1))
                if not j:
                    continue
                pid, nombre, mu, sd = j
                real = None
                if "ps_game" in R:
                    sub = R["ps_game"]
                    v = sub.loc[sub.player_id == pid, POS[pos][0]]
                    real = float(v.iloc[0]) if len(v) else None
                rows.append(Y(f"{pos} {nombre} ({team}) · {etq}", mu, "n", sd, real=real))
    if R:
        rows = [r for r in rows if r["real"] is not None]
    return rows


def nfl():
    import nflreadpy as nfr
    temp = HOY.year if HOY.month >= 3 else HOY.year - 1
    seasons = [temp - 1, temp]
    sch = nfr.load_schedules(seasons).to_pandas()
    sch["gameday"] = pd.to_datetime(sch["gameday"])
    sch["home_team"] = sch["home_team"].replace(NFL_TEAM)
    sch["away_team"] = sch["away_team"].replace(NFL_TEAM)
    jug = sch.dropna(subset=["home_score", "away_score"])
    jugD = jug.rename(columns={"gameday": "Date"})      # para la racha

    try:
        ps, wide, eg_d = stats_nfl(nfr, seasons, sch)
    except Exception as e:
        print("Stats reales NFL no disponibles (solo spread/total):", e)
        ps, wide, eg_d = None, None, {}

    # ── Selección de la semana a mostrar ──
    pend = sch[sch.home_score.isna() & (sch.gameday >= TS(HOY))].sort_values(["season", "week", "gameday"])
    if len(pend):
        s_obj, w0 = int(pend.iloc[0].season), int(pend.iloc[0].week)
        quedan = pend[(pend.season == s_obj) & (pend.week == w0)]
        w_obj = w0
        if len(quedan) <= JUEGOS_MIN_SEMANA:
            sig = pend[(pend.season == s_obj) & (pend.week > w0)]
            if len(sig):
                w_obj = int(sig.week.min())
        prox = pend[(pend.season == s_obj) & (pend.week == w_obj)]
        hoy_p = pend[pend.gameday == TS(HOY)]      # juegos de HOY aún sin marcador: se muestran siempre
        prox = pd.concat([prox, hoy_p]).drop_duplicates(subset=["season", "week", "home_team", "away_team"]) \
                 .sort_values("gameday")
        diag(f"NFL ({HOY}): a la semana {w0} le quedan {len(quedan)} juegos → muestro semana {w_obj} "
             f"+ {len(hoy_p)} juego(s) de hoy")
    else:
        prox = pend

    ult_semana = sch[sch.home_score.notna() & (sch.gameday >= TS(HOY - dt.timedelta(days=7))) &
                     (sch.gameday < TS(HOY))].sort_values("gameday")

    eq = sorted(set(sch.home_team) | set(sch.away_team))
    ix = {t: i for i, t in enumerate(eq)}

    ref_hoy = TS(HOY)
    ref_sem = TS(HOY - dt.timedelta(days=7))
    M_hoy = modelo_nfl(jug, wide, ps, ref_hoy, eq, ix)
    M_sem = modelo_nfl(jug, wide, ps, ref_sem, eq, ix)

    out = []
    for r in prox.itertuples():
        h, a = r.home_team, r.away_team
        mar, tot = pred_nfl(M_hoy, h, a)
        rows = filas_nfl(h, a, mar, tot, r.spread_line, r.total_line, M_hoy, ref_hoy)
        ph = norm_cdf(mar / SD_MARGEN)
        probs = [dict(l=h, p=float(ph), c=ml_dec(getattr(r, "home_moneyline", None))),
                 dict(l=a, p=float(1 - ph), c=ml_dec(getattr(r, "away_moneyline", None)))]
        form = [racha(jugD, h, "home_team", "away_team", "home_score", "away_score", ref_hoy),
                racha(jugD, a, "home_team", "away_team", "home_score", "away_score", ref_hoy)]
        out.append(dict(sp="NFL", lg="NFL", date=str(r.gameday.date()), home=h, away=a,
                        week=int(r.week), div=es_div(r, h, a),
                        logo_h=logo(h, "NFL"), logo_a=logo(a, "NFL"),
                        proy=[round((tot + mar) / 2, 1), round((tot - mar) / 2, 1)],
                        rows=rows, probs=probs, form=form, ia=""))

    hist_res = []
    for r in ult_semana.itertuples():
        h, a = r.home_team, r.away_team
        mar, tot = pred_nfl(M_sem, h, a)
        proy_h, proy_a = round((tot + mar) / 2, 1), round((tot - mar) / 2, 1)
        real_h, real_a = int(r.home_score), int(r.away_score)
        pred_g = h if proy_h > proy_a else a
        real_g = h if real_h > real_a else a
        atino = "SI" if pred_g == real_g else "NO"

        R = dict(spread=real_h - real_a, total=real_h + real_a)
        yh = eg_d.get((int(r.season), int(r.week), h))
        ya = eg_d.get((int(r.season), int(r.week), a))
        if yh:
            R["air_h"], R["gr_h"] = yh
        if ya:
            R["air_a"], R["gr_a"] = ya
        if ps is not None:
            R["ps_game"] = ps[(ps.season == r.season) & (ps.week == r.week)]

        rows = filas_nfl(h, a, mar, tot, r.spread_line, r.total_line, M_sem, ref_sem, R)
        hist_res.append(dict(date=str(r.gameday.date()), home=h, away=a,
                             week=int(r.week), div=es_div(r, h, a),
                             logo_h=logo(h, "NFL"), logo_a=logo(a, "NFL"),
                             proy=[proy_h, proy_a], real=[real_h, real_a], atino=atino, rows=rows))
    return out, hist_res


# ───────────────────────── Fútbol ─────────────────────────
LIGAS = {"La Liga": ("SP1", "esp.1"), "Premier League": ("E0", "eng.1"), "Champions League": (None, "uefa.champions")}
SPORT_KEY = {"La Liga": "soccer_spain_la_liga", "Premier League": "soccer_epl",
             "Champions League": "soccer_uefa_champs_league"}  # claves de The Odds API
EXTRA = ("HC", "AC", "HST", "AST")

# ESPN / casas -> nombre usado en football-data.co.uk (amplía si algún equipo no coincide)
ALIAS = {
    "Manchester United": "Man United", "Manchester City": "Man City", "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest", "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton", "Leeds United": "Leeds", "West Ham United": "West Ham",
    "AFC Bournemouth": "Bournemouth", "Athletic Club": "Ath Bilbao", "Atlético Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid", "Real Sociedad": "Sociedad", "Deportivo Alavés": "Alaves",
    "Rayo Vallecano": "Vallecano", "Celta Vigo": "Celta", "Real Betis": "Betis", "RCD Mallorca": "Mallorca",
    "Real Oviedo": "Oviedo", "Espanyol": "Espanol", "CA Osasuna": "Osasuna",
}


def _n(s):
    s = unicodedata.normalize("NFKD", str(s).lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def mapear(nombre, nombres, estricto=False):
    """Convierte un nombre externo (ESPN / casa de apuestas) al nombre usado en el histórico.
    estricto=True: sin coincidencias por subcadena (evita que 'Real Sociedad B' o 'Real Madrid Castilla'
    se confundan con el primer equipo); si no hay coincidencia segura devuelve None."""
    if nombre in nombres:
        return nombre
    if ALIAS.get(nombre) in nombres:
        return ALIAS[nombre]
    nn = _n(nombre)
    cand = {x: _n(x) for x in nombres}
    sin = lambda t: " ".join(w for w in t.split() if w not in ("fc", "cf", "afc", "ud", "cd", "rcd", "ca", "sad"))
    for x, c in cand.items():
        if c and (c == nn or sin(c) == sin(nn)):
            return x
    if not estricto:
        for x, c in cand.items():
            if c and (c in nn or nn in c):
                return x
    m = difflib.get_close_matches(nn, list(cand.values()), n=1, cutoff=0.85 if estricto else 0.6)
    if m:
        return next(x for x, c in cand.items() if c == m[0])
    return None if estricto else nombre


def cargar_logos(slug, nombres):
    """Escudos desde el endpoint de equipos de ESPN. Se guardan con el nombre de ESPN y con el del histórico."""
    try:
        data = None
        for prm in ({}, {"limit": 200}):
            r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams",
                             params=prm, timeout=30)
            if r.ok:
                data = r.json()
                break
        if data is None:
            raise RuntimeError(f"HTTP {r.status_code}")
        eqs = []
        for sp in data.get("sports", []):
            for lg in sp.get("leagues", []):
                eqs += [t.get("team", {}) for t in lg.get("teams", [])]
        n = 0
        for t in eqs:
            nombre = t.get("displayName")
            url = next((x.get("href") for x in (t.get("logos") or []) if x.get("href")), None) or t.get("logo")
            if not nombre or not url:
                continue
            LOGOS.setdefault(_n(nombre), url)
            m = mapear(nombre, nombres, estricto=True)
            if m:
                LOGOS.setdefault(_n(m), url)
            n += 1
        diag(f"{slug}: {n} escudos cargados desde ESPN")
    except Exception as e:
        diag(f"{slug}: no se pudieron cargar escudos de ESPN ({e})")


TSDB_LIGAS = ["English Premier League", "Spanish La Liga", "UEFA Champions League"]


def _tsdb(ruta, **prm):
    """TheSportsDB (gratis, sin registro). Prueba las llaves públicas de demostración."""
    for k in ("123", "3"):
        try:
            r = requests.get(f"https://www.thesportsdb.com/api/v1/json/{k}/{ruta}", params=prm, timeout=20)
            if r.ok:
                return r.json() or {}
        except Exception:
            pass
    return {}


def _nombres_equipo(t):
    """Nombre del histórico + posibles nombres 'largos' (ESPN) para buscar el escudo."""
    return {t, *[esp for esp, csv in ALIAS.items() if csv == t]}


def logos_externos(faltan):
    """Respaldo: escudos de TheSportsDB para los equipos que ESPN no dio."""
    pool = {}
    for lg in TSDB_LIGAS:
        for t in (_tsdb("search_all_teams.php", l=lg).get("teams") or []):
            url = t.get("strBadge") or t.get("strTeamBadge")
            if not url:
                continue
            for nom in [t.get("strTeam"), t.get("strTeamShort"), *(t.get("strAlternate") or "").split(",")]:
                if nom and nom.strip():
                    pool.setdefault(_n(nom), url)
    n = 0
    for eq in faltan:
        url = next((pool[_n(x)] for x in _nombres_equipo(eq) if _n(x) in pool), None)
        if not url:                                              # búsqueda individual
            for x in sorted(_nombres_equipo(eq), key=len, reverse=True):
                for t in (_tsdb("searchteams.php", t=x).get("teams") or []):
                    u = t.get("strBadge") or t.get("strTeamBadge")
                    if u and t.get("strSport") == "Soccer":
                        url = u
                        break
                if url:
                    break
        if url:
            LOGOS[_n(eq)] = url
            n += 1
    diag(f"TheSportsDB: {n} escudos de respaldo para {len(faltan)} equipos")


def equipos_act(h):
    """Equipos de la temporada en curso: sirve para descartar partidos de otras divisiones o copas."""
    ini = HOY.year if HOY.month >= 7 else HOY.year - 1
    d = h[h.Date >= TS(dt.date(ini, 7, 1))]
    act = set(d.h) | set(d.a)
    return act if len(act) >= 18 else None     # si aún no hay jornadas suficientes, no se filtra


def temp_fd(off=0):
    y = (HOY.year if HOY.month >= 7 else HOY.year - 1) + off
    return f"{str(y)[2:]}{str(y + 1)[2:]}"


def fechas(s):
    """Fechas dd/mm/aa o dd/mm/aaaa (incluso mezcladas en el mismo archivo)."""
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s, dayfirst=True, errors="coerce")


def poisson_1x2(lh, la, n=11):
    pm = lambda l: np.array([math.exp(-l) * l ** i / math.factorial(i) for i in range(n)])
    ph, pa = pm(lh), pm(la)
    G = np.outer(ph, pa)
    return np.tril(G, -1).sum(), np.trace(G), np.triu(G, 1).sum(), 1 - ph[0] - pa[0] + ph[0] * pa[0]


def stat_espn(c, nombres):
    for s in c.get("statistics", []) or []:
        if s.get("name") in nombres:
            return num(s.get("displayValue"))
    return None


def _espn_dia(slug, d):
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
    r = requests.get(url, params={"dates": f"{d:%Y%m%d}"}, timeout=30)
    r.raise_for_status()
    return r.json().get("events", [])


def _espn_ventana(slug, d1, d2):
    """ESPN ya no acepta rangos de fechas (error 400): se pide un día a la vez, en paralelo."""
    dias = [d1 + dt.timedelta(days=i) for i in range((d2 - d1).days + 1)]
    fallos = []

    def uno(d):
        try:
            return _espn_dia(slug, d)
        except Exception as e:
            fallos.append(str(e))
            return []

    with ThreadPoolExecutor(8) as ex:
        eventos = [e for lst in ex.map(uno, dias) for e in lst]
    if fallos and len(fallos) == len(dias):
        raise RuntimeError(f"ESPN falló en los {len(dias)} días ({fallos[0]})")

    out = []
    for e in eventos:
        try:
            c = e["competitions"][0]
            H = next(x for x in c["competitors"] if x["homeAway"] == "home")
            Av = next(x for x in c["competitors"] if x["homeAway"] == "away")
            for T in (H, Av):
                if T["team"].get("logo"):
                    LOGOS.setdefault(_n(T["team"]["displayName"]), T["team"]["logo"])
            out.append(dict(Date=pd.to_datetime(e["date"], utc=True).tz_convert(None),
                            h=H["team"]["displayName"], a=Av["team"]["displayName"],
                            FTHG=num(H.get("score")), FTAG=num(Av.get("score")),
                            HC=stat_espn(H, ("wonCorners",)), AC=stat_espn(Av, ("wonCorners",)),
                            HST=stat_espn(H, ("shotsOnTarget",)), AST=stat_espn(Av, ("shotsOnTarget",)),
                            fin=bool(e["status"]["type"]["completed"])))
        except Exception:
            continue
    return pd.DataFrame(out)


def espn(slug, d1, d2, paso=30):
    """ESPN falla con rangos largos de fechas: se consulta en ventanas de ~30 días."""
    frames, d = [], d1
    while d <= d2:
        e2 = min(d + dt.timedelta(days=paso - 1), d2)
        try:
            f = _espn_ventana(slug, d, e2)
            if len(f):
                frames.append(f)
        except Exception as e:
            diag(f"ESPN {slug} {d:%Y-%m-%d}→{e2:%Y-%m-%d}: {e}")
        d = e2 + dt.timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Date", "h", "a"])


def espn_prox(slug, nombres=None, act=None):
    """Próximos partidos de ESPN: busca hacia adelante en ventanas de ESPN_PASO días
    hasta juntar N_PROX (o agotar ESPN_DIAS). Así no depende de que haya partidos esta semana
    (pausas FIFA, jornadas lejanas, etc.)."""
    res, vistos, ajenos, d = [], set(), set(), HOY
    limite = HOY + dt.timedelta(days=ESPN_DIAS)
    while d <= limite and len(res) < N_PROX:
        e2 = min(d + dt.timedelta(days=ESPN_PASO - 1), limite)
        try:
            f = _espn_ventana(slug, d, e2)
            if len(f):
                f = f[~f.fin & (f.Date >= TS(HOY))].sort_values("Date")
                for r in f.itertuples():
                    strict = bool(act and nombres)
                    h2 = mapear(r.h, nombres, estricto=strict) if nombres else r.h
                    a2 = mapear(r.a, nombres, estricto=strict) if nombres else r.a
                    if act and (h2 is None or a2 is None or h2 not in act or a2 not in act):
                        ajenos.add(f"{r.h}-{r.a}")
                        continue
                    if (h2, a2) in vistos:
                        continue
                    vistos.add((h2, a2))
                    res.append(dict(h=h2, a=a2, Date=r.Date, cu=[None] * 3, src="ESPN"))
        except Exception as e:
            diag(f"ESPN {slug} {d:%Y-%m-%d}→{e2:%Y-%m-%d}: {e}")
        d = e2 + dt.timedelta(days=1)
    if ajenos:
        diag(f"ESPN {slug}: descartados {len(ajenos)} partidos de equipos ajenos a la liga, ej. {sorted(ajenos)[:3]}")
    diag(f"ESPN {slug}: {len(res)} próximos encontrados (hasta {limite:%Y-%m-%d})")
    return sorted(res, key=lambda x: x["Date"])[:N_PROX]


FD_COD = {"La Liga": "PD", "Premier League": "PL", "Champions League": "CL"}


def fd_org(liga, nombres=None, act=None):
    """OPCIONAL: próximos partidos desde football-data.org (secreto FOOTBALL_DATA_KEY, plan gratis)."""
    k = secreto("FOOTBALL_DATA_KEY")
    code = FD_COD.get(liga)
    if not k or not code:
        diag(f"{liga}: football-data.org omitido (falta el secreto FOOTBALL_DATA_KEY)")
        return []
    try:
        r = requests.get(f"https://api.football-data.org/v4/competitions/{code}/matches",
                         headers={"X-Auth-Token": k}, params=dict(status="SCHEDULED,TIMED"), timeout=30)
        r.raise_for_status()
        res = []
        for m in r.json().get("matches", []):
            ht, at = m.get("homeTeam") or {}, m.get("awayTeam") or {}
            if not ht.get("name") or not at.get("name"):      # llaves por definir (eliminatorias)
                continue
            n1, n2 = ht.get("shortName") or ht["name"], at.get("shortName") or at["name"]
            f = pd.to_datetime(m["utcDate"], utc=True).tz_convert(None)
            if f < TS(HOY):
                continue
            strict = bool(act and nombres)
            h2 = mapear(n1, nombres, estricto=strict) if nombres else n1
            a2 = mapear(n2, nombres, estricto=strict) if nombres else n2
            if act and (h2 is None or a2 is None or h2 not in act or a2 not in act):
                continue
            for T, nm in ((ht, h2), (at, a2)):
                if T.get("crest"):
                    LOGOS.setdefault(_n(nm), T["crest"])
            res.append(dict(h=h2, a=a2, Date=f, cu=[None] * 3, src="football-data.org"))
        res = sorted(res, key=lambda x: x["Date"])[:N_PROX]
        diag(f"{liga}: football-data.org devolvió {len(res)} próximos")
        return res
    except Exception as e:
        diag(f"{liga}: football-data.org falló ({e})")
        return []


def completar_espn(prox, hist, slug, liga, act=None):
    """Si faltan partidos para llegar a N_PROX, completa con football-data.org y luego ESPN."""
    nombres = sorted(set(hist.h) | set(hist.a))
    vistos = {(p["h"], p["a"]) for p in prox}
    extra = fd_org(liga, nombres, act)
    if len(prox) + len(extra) < N_PROX:
        extra = extra + espn_prox(slug, nombres, act)
    for p in extra:
        if len(prox) >= N_PROX:
            break
        if (p["h"], p["a"]) in vistos:
            continue
        vistos.add((p["h"], p["a"]))
        prox.append(p)
    return sorted(prox, key=lambda x: x["Date"])[:N_PROX]


def cuotas_fila(r):
    """Primer juego de cuotas (decimales) completo y válido en la fila de fixtures.csv."""
    for cols in (("AvgH", "AvgD", "AvgA"), ("MaxH", "MaxD", "MaxA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA")):
        v = [num(r.get(c)) for c in cols]
        if all(x is not None for x in v):
            return v
    return [None] * 3


def fuentes_csv(liga, div, slug):
    hs = []
    for off in (0, -1):
        try:
            df_csv = pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{temp_fd(off)}/{div}.csv",
                                 encoding="latin-1", on_bad_lines="skip")
            if "Div" in df_csv.columns:
                df_csv = df_csv[df_csv.Div == div]
            hs.append(df_csv)
        except Exception as e:
            diag(f"{liga}: no se pudo leer {temp_fd(off)}/{div}.csv ({e})")
    h = pd.concat(hs).dropna(subset=["FTHG", "FTAG"])
    h = h.rename(columns={"HomeTeam": "h", "AwayTeam": "a"})
    h = h.dropna(subset=["h", "a"])
    h["Date"] = fechas(h.Date)
    h = h.dropna(subset=["Date"])
    for c in EXTRA:
        h[c] = pd.to_numeric(h[c], errors="coerce") if c in h.columns else np.nan

    act = equipos_act(h)
    prox = []
    try:
        f = pd.read_csv("https://www.football-data.co.uk/fixtures.csv", encoding="latin-1", on_bad_lines="skip")
        if "Div" in f.columns:
            f = f[f.Div == div].copy()
        f["Date"] = fechas(f.Date)
        diag(f"{liga}: fixtures.csv trae {len(f)} partidos de la división")
        f = f[f.Date >= TS(HOY)].sort_values("Date")
        prox = [dict(h=r["HomeTeam"], a=r["AwayTeam"], Date=r["Date"], cu=cuotas_fila(r), src="fixtures.csv")
                for r in f.to_dict("records")]
        if act:
            ajenos = [p for p in prox if p["h"] not in act or p["a"] not in act]
            if ajenos:
                diag(f"{liga}: descartados {len(ajenos)} partidos de equipos ajenos a la liga, "
                     f"ej. {[p['h'] + '-' + p['a'] for p in ajenos[:3]]}")
            prox = [p for p in prox if p["h"] in act and p["a"] in act]
        prox = prox[:N_PROX]
    except Exception as e:
        diag(f"{liga}: fixtures.csv falló ({e})")
    if len(prox) < N_PROX:
        prox = completar_espn(prox, h, slug, liga, act)
    return h, prox


def fuentes_espn(liga, slug):
    ini = HOY.year if HOY.month >= 7 else HOY.year - 1
    d = espn(slug, dt.date(ini, 8, 1), HOY)    # solo la temporada actual (una petición por día, en paralelo)
    if len(d) == 0:
        diag(f"{liga}: ESPN no devolvió histórico")
        return d, []
    h = d[d.fin].dropna(subset=["FTHG", "FTAG"]).copy()
    h["Date"] = pd.to_datetime(h.Date)
    for c in EXTRA:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    nombres = sorted(set(h.h) | set(h.a))
    act = None if liga == "Champions League" else equipos_act(h)
    return h, (fd_org(liga, nombres, act) or espn_prox(slug, nombres, act))


def fuentes(liga, div, slug):
    if div:
        try:
            h, prox = fuentes_csv(liga, div, slug)
            if len(h) > 0:
                return h, prox
            diag(f"{liga}: el CSV no trae histórico, uso ESPN")
        except Exception as e:
            diag(f"{liga}: CSV falló ({e}), uso ESPN")
    return fuentes_espn(liga, slug)


def momios_api(sport, nombres):
    """OPCIONAL: cuotas promedio 1X2 desde The Odds API (secreto ODDS_API_KEY, plan gratis).
    Sirve sobre todo para Champions, que no trae cuotas en ESPN."""
    k = secreto("ODDS_API_KEY")
    if not k:
        return {}
    try:
        r = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport}/odds/",
                         params=dict(apiKey=k, regions="eu,uk", markets="h2h", oddsFormat="decimal"), timeout=30)
        r.raise_for_status()
        out = {}
        for ev in r.json():
            ac = {"H": [], "D": [], "A": []}
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk.get("key") != "h2h":
                        continue
                    for o in mk.get("outcomes", []):
                        if o["name"] == ev["home_team"]:
                            ac["H"].append(o["price"])
                        elif o["name"] == ev["away_team"]:
                            ac["A"].append(o["price"])
                        elif o["name"] == "Draw":
                            ac["D"].append(o["price"])
            if all(ac.values()):
                key = (mapear(ev["home_team"], nombres), mapear(ev["away_team"], nombres))
                out[key] = [round(float(np.mean(ac[x])), 3) for x in "HDA"]
        diag(f"{sport}: The Odds API devolvió cuotas de {len(out)} partidos")
        return out
    except Exception as e:
        diag(f"The Odds API ({sport}) falló: {e}")
        return {}


def modelo_opc(df, vh, va, ref, minimo=40):
    if vh in df.columns and va in df.columns:
        d = df.dropna(subset=[vh, va])
        if len(d) >= minimo:
            return tasas(d, vh, va, ref=ref)
    return None


def lam_total(m, h, a, fallback):
    if m is None:
        return fallback
    x, y = esperado(m, h, a)
    return x + y


def futbol():
    out, hist_res = [], []
    corte = TS(HOY - dt.timedelta(days=DIAS_RES))
    for liga, (div, slug) in LIGAS.items():
        try:
            hist, prox = fuentes(liga, div, slug)
            diag(f"{liga}: histórico={len(hist)} partidos, próximos={len(prox)}")
            if len(hist) == 0:
                continue
            ref_hoy = TS(HOY)
            hp = hist[hist.Date < corte]
            usar_prev = len(hp) > 20

            tg = tasas(hist, "FTHG", "FTAG", ref=ref_hoy)
            tc = modelo_opc(hist, "HC", "AC", ref_hoy)
            tt = modelo_opc(hist, "HST", "AST", ref_hoy)
            if usar_prev:
                tg_p = tasas(hp, "FTHG", "FTAG", ref=corte)
                tc_p = modelo_opc(hp, "HC", "AC", corte)
                tt_p = modelo_opc(hp, "HST", "AST", corte)
            else:
                tg_p, tc_p, tt_p = tg, tc, tt
            prox = sorted(prox, key=lambda x: x["Date"])[:N_PROX]
            for p in prox:
                diag(f"{liga} próximo: {p['Date'].date()} {p['h']} vs {p['a']} [{p.get('src', '?')}]")
            nombres = sorted(set(hist.h) | set(hist.a))
            odds = momios_api(SPORT_KEY[liga], nombres)
            cargar_logos(slug, nombres)

            # ── Resultados de la última semana ──
            sem = hist[(hist.Date >= corte) & (hist.Date < ref_hoy)].sort_values("Date")
            for r in sem.itertuples():
                try:
                    lh, la = esperado(tg_p, r.h, r.a)
                    _, _, _, btts = poisson_1x2(lh, la)
                    real_h, real_a = int(r.FTHG), int(r.FTAG)
                    pred_g = "H" if lh > la + 0.1 else ("A" if la > lh + 0.1 else "D")
                    real_g = "H" if real_h > real_a else ("A" if real_a > real_h else "D")
                    atino = "SI" if pred_g == real_g else "NO"

                    ce = lam_total(tc_p, r.h, r.a, 8.5 + (lh + la) * 0.4)
                    tp = lam_total(tt_p, r.h, r.a, 7.0 + (lh + la) * 0.8)
                    hc, ac = num(getattr(r, "HC", None)), num(getattr(r, "AC", None))
                    hst, ast = num(getattr(r, "HST", None)), num(getattr(r, "AST", None))

                    rows_res = [
                        fila("Goles totales", lh + la, "p", real=real_h + real_a),
                        fila("Ambos anotan", btts, "p", o="Sí", u="No", real=1 if (real_h > 0 and real_a > 0) else 0),
                        fila("Tiros de esquina", ce, "p", real=(hc + ac) if hc is not None and ac is not None else None),
                        fila("Tiros a puerta totales", tp, "p", real=(hst + ast) if hst is not None and ast is not None else None),
                    ]
                    rows_res = [x for x in rows_res if x["real"] is not None]
                    hist_res.append(dict(liga=liga, date=str(r.Date.date()), home=r.h, away=r.a,
                                         logo_h=logo(r.h, "FUT"), logo_a=logo(r.a, "FUT"),
                                         proy=[round(lh, 1), round(la, 1)], real=[real_h, real_a],
                                         atino=atino, rows=rows_res))
                except Exception as e:
                    diag(f"{liga} resultado {r.h}-{r.a}: {e}")

            # ── Próximos partidos ──
            con_cuota = 0
            for p in prox:
                try:
                    h, a = p["h"], p["a"]
                    lh, la = esperado(tg, h, a)
                    p1, px, p2, btts = poisson_1x2(lh, la)
                    ce = lam_total(tc, h, a, 8.5 + (lh + la) * 0.4)
                    tp = lam_total(tt, h, a, 7.0 + (lh + la) * 0.8)
                    rows = [
                        fila("Goles totales", lh + la, "p", None, 2.5),
                        fila("Tiros de esquina", ce, "p", None, 9.5),
                        fila("Ambos anotan", btts, "p", p=btts, o="Sí", u="No"),
                        fila("Tiros a puerta totales", tp, "p", None, 8.5),
                    ]
                    cu = p["cu"]
                    if not any(cu):
                        cu = odds.get((h, a), cu)
                    con_cuota += 1 if all(cu) else 0
                    probs = [dict(l=l, p=float(pp), c=c) for l, pp, c in zip((h, "Empate", a), (p1, px, p2), cu)]
                    form = [racha(hist, h, "h", "a", "FTHG", "FTAG", ref_hoy),
                            racha(hist, a, "h", "a", "FTHG", "FTAG", ref_hoy)]
                    out.append(dict(sp="FUT", lg=liga, date=str(p["Date"].date()), home=h, away=a,
                                    logo_h=logo(h, "FUT"), logo_a=logo(a, "FUT"),
                                    proy=[round(lh, 2), round(la, 2)], rows=rows, probs=probs, form=form, ia=""))
                except Exception as e:
                    diag(f"{liga} próximo {p.get('h')}-{p.get('a')}: {e}")
            diag(f"{liga}: {con_cuota} de {len(prox)} partidos con momios 1X2")
        except Exception as e:
            diag(f"[{liga}] omitida: {type(e).__name__}: {e}")

    # Escudos: lo que ESPN no dio se busca en TheSportsDB; después se asigna a cada tarjeta
    usados = sorted({t for g in out + hist_res for t in (g["home"], g["away"])})
    faltan = [t for t in usados if not LOGOS.get(_n(t))]
    if faltan:
        try:
            logos_externos(faltan)
        except Exception as e:
            diag(f"Escudos de respaldo fallaron: {e}")
    for g in out + hist_res:
        g["logo_h"], g["logo_a"] = logo(g["home"], "FUT"), logo(g["away"], "FUT")
    sin = [t for t in usados if not LOGOS.get(_n(t))]
    diag(f"Escudos: {len(usados) - len(sin)} de {len(usados)} equipos" + (f"; sin escudo: {sin}" if sin else ""))
    return out, hist_res


# ───────────────────────── HTML (interfaz nueva) ─────────────────────────
HTML_PART1 = r"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Edge Lab"><meta name="theme-color" content="#090912">
<title>Edge Lab · Apuestas</title>
<link rel="icon" href="icon.png">
<link rel="apple-touch-icon" href="icon.png">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#090912;--card:#11112a;--line:#2a1b3a;--red:#d6264f;--tx:#e9e9f2;--mut:#8a8aa3;--ok:#3c7a4a;--wrong:#e03e3e}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html{-webkit-text-size-adjust:100%;text-size-adjust:100%}
body{margin:0;overflow-x:hidden;overflow-x:clip;min-height:100vh;min-height:100dvh;background:radial-gradient(1200px 500px at 50% -100px,#2a0f26 0,transparent 70%),var(--bg);color:var(--tx);font-family:Barlow,-apple-system,sans-serif;padding-bottom:env(safe-area-inset-bottom)}
header.hd{position:-webkit-sticky;position:sticky;top:0;z-index:50;background:rgba(9,9,18,.88);-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);padding-top:env(safe-area-inset-top)}
/* ticker */
.tick{overflow:hidden;border-bottom:1px solid #1d1530;background:#0c0c1e;height:34px;display:flex;align-items:center}
.tick .lbl{flex:none;background:var(--red);color:#fff;font:800 12px 'Barlow Condensed';letter-spacing:.16em;padding:0 12px;height:100%;display:flex;align-items:center;z-index:2}
.trk{overflow:hidden;flex:1;white-space:nowrap}
.tk{display:inline-block;white-space:nowrap;will-change:transform;-webkit-animation:mv var(--d,60s) linear infinite;animation:mv var(--d,60s) linear infinite}
@-webkit-keyframes mv{from{-webkit-transform:translateX(0)}to{-webkit-transform:translateX(-50%)}}
@keyframes mv{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.ti{display:inline-flex;align-items:center;gap:6px;margin-right:34px;font:700 14px 'Barlow Condensed';letter-spacing:.06em;text-transform:uppercase;vertical-align:middle}
.ti .lg2{color:var(--red);font-size:11px;letter-spacing:.16em}.ti img{width:18px;height:18px;object-fit:contain}
.ti em{font-style:normal;color:#fff;font-size:15px}.ti .ok{color:#6fd08a}.ti .no{color:var(--wrong)}
@media(prefers-reduced-motion:reduce){.tk{-webkit-animation:none;animation:none}.trk{overflow-x:auto}}
nav{display:flex;align-items:center;gap:30px;padding:12px 40px;border-bottom:1px solid #1d1530;font-family:'Barlow Condensed';letter-spacing:.14em;font-weight:700;overflow-x:auto;white-space:nowrap;-webkit-overflow-scrolling:touch;scrollbar-width:none}
nav::-webkit-scrollbar{display:none}
nav a{cursor:pointer;color:var(--mut);font-size:15px;padding:8px 0;transition:.2s;flex:none}nav a.on{color:var(--red);border-bottom:2px solid var(--red)}
.fmt{margin-left:auto;flex:none;background:transparent;border:1px solid var(--red);color:var(--red);border-radius:20px;padding:8px 16px;cursor:pointer;font:700 13px 'Barlow Condensed';letter-spacing:.14em;text-transform:uppercase;-webkit-appearance:none}
.fmt:hover{background:var(--red);color:#fff}
main{padding:30px 40px;max-width:1500px;margin:0 auto}
h1{font:800 56px 'Barlow Condensed';margin:0;text-transform:uppercase;line-height:1}h1 span,h1 i{color:var(--red);font-style:italic}
.sub{color:var(--mut);margin:8px 0 24px}
/* lobby */
.lobby{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:20px}
.tile{position:relative;overflow:hidden;border-radius:20px;border:1px solid var(--line);min-height:210px;padding:26px;cursor:pointer;display:flex;flex-direction:column;justify-content:flex-end;transition:transform .2s,border-color .2s;-webkit-user-select:none;user-select:none}
.tile:hover{transform:translateY(-4px);border-color:var(--red)}.tile:active{transform:scale(.98)}
.t-NFL{background:linear-gradient(135deg,#0f2a1e,#11112a 70%)}.t-FUT{background:linear-gradient(135deg,#1b0f2e,#2a0f26)}
.t-CL{background:linear-gradient(135deg,#0f1a3a,#1b0f2e)}.t-RES{background:linear-gradient(135deg,#3a1226,#11112a 70%)}
.tbg{position:absolute;inset:0;pointer-events:none}
.tbg img{position:absolute;width:110px;height:110px;object-fit:contain;opacity:.16;filter:drop-shadow(0 0 20px rgba(0,0,0,.5))}
.tbg img:nth-child(1){right:-10px;top:-10px;transform:rotate(12deg)}.tbg img:nth-child(2){right:90px;top:50px;transform:rotate(-8deg);width:90px;height:90px}
.tbg img:nth-child(3){right:10px;top:110px;transform:rotate(6deg);width:80px;height:80px}.tbg img:nth-child(4){right:110px;top:-20px;width:70px;height:70px;transform:rotate(-14deg)}
.tic{position:absolute;top:18px;left:22px;font-size:46px;line-height:1;filter:drop-shadow(0 4px 12px rgba(0,0,0,.5))}
.tile h2{position:relative;margin:0;font:800 40px 'Barlow Condensed';text-transform:uppercase;letter-spacing:.04em}
.tile p{position:relative;margin:4px 0 10px;color:#c9c9dc;font-size:14px}
.tile .nx{position:relative;color:var(--mut);font-size:12px;margin-bottom:12px;min-height:15px}
.go{position:relative;align-self:flex-start;font:800 13px 'Barlow Condensed';letter-spacing:.16em;color:var(--red);border:1px solid var(--red);border-radius:20px;padding:5px 14px}
.tile:hover .go{background:var(--red);color:#fff}
.chips{display:flex;gap:10px;margin-bottom:22px;flex-wrap:wrap}
.chip{border:1px solid var(--line);border-radius:20px;padding:8px 16px;cursor:pointer;font:700 13px 'Barlow Condensed';letter-spacing:.14em;color:var(--mut);text-transform:uppercase}
.chip.on{border-color:var(--red);color:var(--red)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,400px),1fr));gap:20px}
.card{min-width:0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 24px;transition:.2s}.card:hover{border-color:var(--red)}
.top{display:flex;justify-content:space-between;align-items:center}.lg{font:700 14px 'Barlow Condensed';letter-spacing:.18em;color:#c9c9dc;text-transform:uppercase}
.badge{background:var(--ok);font:800 13px 'Barlow Condensed';letter-spacing:.12em;border-radius:14px;padding:3px 12px}
.badge.incorrecto{background:var(--wrong)}
.val{font:800 12px 'Barlow Condensed';letter-spacing:.12em;color:#6fd08a;margin-right:8px}
.fecha{color:var(--red);font:700 13px 'Barlow Condensed';letter-spacing:.16em;margin:6px 0 18px;text-transform:uppercase}
.vs{display:flex;align-items:center;justify-content:space-between;gap:8px}.eq{text-align:center;flex:1;min-width:0}
.ci{width:64px;height:64px;border-radius:50%;background:linear-gradient(135deg,#1b0f2e,#11112a);border:1px solid var(--line);margin:0 auto 8px;display:grid;place-items:center;overflow:hidden;font:800 20px 'Barlow Condensed';color:var(--mut)}
.ci img{width:80%;height:80%;object-fit:contain}
.eq b{font:700 18px 'Barlow Condensed';text-transform:uppercase;display:block;margin-top:4px}.sc{font:800 44px 'Barlow Condensed';white-space:nowrap}
/* racha últimos 5 */
.fm{display:flex;gap:4px;justify-content:center;margin-top:6px}
.fm i{width:7px;height:7px;border-radius:50%;background:#555570}
.fm i.G{background:#6fd08a}.fm i.E{background:#b8861b}.fm i.P{background:#e03e3e}
.scw{text-align:center}.pj{font:700 12px 'Barlow Condensed';letter-spacing:.14em;color:var(--mut);text-transform:uppercase;white-space:nowrap}
.pj.p{color:#c9c9dc;margin-bottom:2px}
.pr{display:flex;height:6px;border-radius:3px;overflow:hidden;margin:16px 0 4px}
.pl{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);gap:6px}
/* desplegable de momios */
.mbtn{width:100%;max-width:100%;margin-top:16px;display:flex;justify-content:space-between;align-items:center;background:#0e0e24;border:1px solid var(--line);border-radius:10px;color:var(--tx);padding:12px 14px;cursor:pointer;font:800 13px 'Barlow Condensed';letter-spacing:.16em;text-transform:uppercase;-webkit-appearance:none;appearance:none}
.mbtn:hover{border-color:var(--red)}
.ar{display:inline-block;color:var(--red);font-size:16px;transition:transform .25s}
.mx.open .ar{transform:rotate(180deg)}.mx.open .mbtn{border-color:var(--red)}
.mbody{display:none}.mx.open .mbody{display:block}
.edge{overflow:hidden;margin-top:14px;border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:#0e0e24}
.edge h4{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 8px;font:800 13px 'Barlow Condensed';letter-spacing:.16em;color:var(--red)}
.fmt2{flex:none;background:transparent;border:1px solid var(--red);color:var(--red);border-radius:16px;padding:5px 12px;cursor:pointer;font:700 12px 'Barlow Condensed';letter-spacing:.12em;text-transform:uppercase;-webkit-appearance:none;appearance:none}
.fmt2:active{background:var(--red);color:#fff}
.eo{display:grid;grid-template-columns:minmax(0,1fr) auto 70px 64px;gap:8px;align-items:center;font-size:13px;padding:3px 0}.eo>*{min-width:0}.eo>span:first-child{overflow-wrap:anywhere}
.eo .fj{color:var(--mut);font-size:12px;white-space:nowrap}.eo .ed{font-weight:700;text-align:right}
table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px}td{padding:6px 4px;border-top:1px solid #1d1d3a}td:nth-child(2){color:var(--mut)}
input{width:62px;background:#0c0c1e;border:1px solid var(--line);border-radius:6px;color:var(--tx);padding:5px 6px;-webkit-appearance:none;appearance:none}
.eo input{width:70px}
.g{color:#6fd08a;font-weight:500}
.ctx{margin-top:12px;color:var(--mut);font-size:12px;border-left:2px solid var(--red);padding-left:10px}
.ai{border:1px solid #6b2a5a;background:linear-gradient(135deg,#1b0f2e,#2a0f26);border-radius:14px;padding:18px 22px;margin-bottom:22px;line-height:1.55;color:#e9d5ff}.ai h3{margin:0 0 6px;font:800 16px 'Barlow Condensed';letter-spacing:.16em;color:var(--red)}
.pks{border:1px solid #6b2a5a;background:linear-gradient(135deg,#1b0f2e,#2a0f26);border-radius:14px;padding:18px 22px;margin-bottom:22px}
.pks h3{margin:0 0 8px;font:800 16px 'Barlow Condensed';letter-spacing:.16em;color:var(--red)}
.pk{display:flex;gap:12px;align-items:flex-start;padding:10px 0;border-top:1px solid #3a1f4a;font-size:14px;line-height:1.45}
.pk .cf{font:800 12px 'Barlow Condensed';letter-spacing:.12em;border-radius:12px;padding:2px 10px;white-space:nowrap;background:#555570}
.pk .cf.alta{background:var(--ok)}.pk .cf.media{background:#b8861b}
.pk small{color:var(--mut);display:block}.pk b{color:#fff}
.tabs{display:flex;gap:30px;flex:none}.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}.tw table{margin-top:12px}td{overflow-wrap:anywhere}
.tag{background:#2a1b3a;color:#e9d5ff;border-radius:8px;padding:1px 8px;margin-left:6px;font-size:11px}
.diag{color:var(--mut);font-size:12px;line-height:1.6;margin-top:10px;white-space:pre-wrap}
.nota{color:var(--mut);font-size:12px;margin-top:26px}
@media(pointer:coarse){input,.eo input{font-size:16px}}   /* evita el zoom automático de Safari al escribir */
@media(max-width:640px){
 main{padding:20px 16px calc(24px + env(safe-area-inset-bottom))}nav{padding:10px 16px;gap:20px}h1{font-size:40px}
 .card{padding:18px 16px}.sc{font-size:34px}.eq b{font-size:15px}.ci{width:52px;height:52px}
 nav{flex-wrap:wrap;overflow:visible;row-gap:6px}.fmt{order:2;padding:6px 12px}
 .tabs{order:1;width:100%;gap:22px;overflow-x:auto;-webkit-overflow-scrolling:touch;scrollbar-width:none}.tabs::-webkit-scrollbar{display:none}
 .eo{grid-template-columns:minmax(0,1fr) 84px 58px;gap:6px}.eo .fj{grid-column:1/-1;order:3;white-space:normal}.eo input{width:84px}
 table{font-size:12px}td{padding:6px 3px}.g{font-size:11px}
 .tile{min-height:180px;padding:20px}.tile h2{font-size:34px}
}
</style></head><body>
<header class="hd">
<div class="tick" id="tick"></div>
<nav>
<div class="tabs"><a id="tHOME" onclick="setTab('HOME')">⌂ HOME</a>
<a id="tNFL" onclick="setTab('NFL')">NFL</a>
<a id="tFUT" onclick="setTab('FUT')">FÚTBOL</a>
<a id="tCL" onclick="setTab('CL')">CHAMPIONS</a>
<a id="tRES" onclick="setTab('RES')">RESULTADOS</a></div>
<button class="fmt" id="fmtbtn" onclick="toggleFmt()"></button>
</nav></header>
<main><h1 id="ttl"></h1><div id="descr" class="sub"></div>
<div id="home" class="lobby"></div>
<div id="ai"></div><div id="picks"></div><div class="chips" id="chips"></div><div class="grid" id="grid"></div><div class="nota">Modelo estadístico (Ridge / Poisson / Normal). Edge = probabilidad del modelo × cuota decimal − 1 (el momio de la casa ya incluye su margen). Puedes escribir tu momio en americano (+150, -110) o decimal (2.50): se detecta solo. 🏠 = equipo local. Puntos bajo cada equipo = racha de sus últimos 5 partidos (verde ganó, ámbar empató, rojo perdió; el más reciente a la derecha). Los picks de Gemini combinan tu modelo con noticias de la web y pueden fallar. Solo informativo, no es asesoría financiera. En Resultados se visualizan exclusivamente filas con datos reales.</div></main>"""


HTML_PART2_JS = r"""<script>
const D = __DATA__;
const HASH = {home:'HOME', nfl:'NFL', fut:'FUT', cl:'CL', res:'RES'};
let tab = HASH[(location.hash || '').slice(1)] || 'HOME', lg = 'Todas', FMT = 'us';
const ROWS = [], ROWK = [], PR = [], PRK = [], CK = [];
const ODDS = {}, LINES = {}, OPEN = new Set();
const TT = {NFL:'NFL', FUT:'Fútbol', CL:'Champions', RES:'Resultados'};
const SUB = {NFL:'Semana actual · modelo vs. mercado', FUT:'La Liga y Premier League', CL:'UEFA Champions League', RES:'Lo que proyectó el modelo vs. lo que pasó realmente'};

const toUS = d => d >= 2 ? '+' + Math.round((d - 1) * 100) : '-' + Math.round(100 / (d - 1));
const fmtO = d => (d == null || !isFinite(d) || d <= 1) ? '' : (FMT === 'us' ? toUS(d) : d.toFixed(2));
function parseO(s) {
  s = String(s).trim().replace(',', '.');
  if (s === '') return null;
  const v = parseFloat(s);
  if (isNaN(v)) return null;
  if (Math.abs(v) >= 100) return v > 0 ? 1 + v / 100 : 1 + 100 / Math.abs(v);
  return v > 1 ? v : null;
}
function toggleFmt() { FMT = FMT === 'us' ? 'dec' : 'us'; render(); }

const erf = x => {
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
};
const cdf = z => 0.5 * (1 + erf(z / Math.SQRT2));
const pcdf = (k, l) => { let s = 0, t = Math.exp(-l); for (let i = 0; i <= k; i++) { s += t; t *= l / (i + 1); } return s; };

function pOver(r, line) {
  if (r.p != null) return r.p;
  if (line === '' || line == null || isNaN(line)) return null;
  return r.d === 'n' ? 1 - cdf((line - r.m) / r.sd) : 1 - pcdf(Math.floor(line), r.m);
}
function txt(r, line) {
  let p = pOver(r, line);
  if (p == null) return '';
  p = Math.min(0.99, Math.max(0.01, p));
  return `${r.o} ${(p * 100).toFixed(0)}% @${fmtO(1 / p)} · ${r.u} ${((1 - p) * 100).toFixed(0)}% @${fmtO(1 / (1 - p))}`;
}
function upd(i) {
  const v = document.getElementById('i' + i).value;
  LINES[ROWK[i]] = v;
  document.getElementById('o' + i).textContent = v === '' ? '' : txt(ROWS[i], parseFloat(v));
}
function edge(gid) {
  let best = null;
  PR[gid].forEach((x, k) => {
    const key = PRK[gid] + '|' + k;
    const d = key in ODDS ? ODDS[key] : x.c;
    const el = document.getElementById(`e${gid}_${k}`);
    if (d == null || d <= 1) { el.innerHTML = ''; return; }
    const e = x.p * d - 1;
    if (best == null || e > best) best = e;
    el.innerHTML = `<span style="color:${e >= 0 ? '#6fd08a' : '#e03e3e'}">${e >= 0 ? '+' : ''}${(e * 100).toFixed(1)}%</span>`;
  });
  const b = document.getElementById('b' + gid);
  if (b) b.textContent = (best != null && best > 0) ? `VALOR +${(best * 100).toFixed(1)}%` : '';
}
function setOdd(gid, k, val) { ODDS[PRK[gid] + '|' + k] = parseO(val); edge(gid); }

function rowH(r, hist, mk) {
  const i = ROWS.push(r) - 1;
  ROWK[i] = mk + '|' + r.l;
  if (hist) return `<tr><td>${r.l}</td><td>${r.m}</td><td>${r.real ?? '–'}</td></tr>`;
  const lv = ROWK[i] in LINES ? LINES[ROWK[i]] : (r.line ?? '');
  const inp = r.p != null ? '' : `<input id="i${i}" type="number" inputmode="decimal" step="0.5" value="${lv}" oninput="upd(${i})">`;
  return `<tr><td>${r.l}</td><td>${r.m}</td><td>${inp}</td><td class="g" id="o${i}">${txt(r, lv === '' ? null : parseFloat(lv))}</td></tr>`;
}
const ini = n => n.split(/\s+/).map(w => w[0]).join('').slice(0, 3).toUpperCase();
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// racha de los últimos 5: G ganó, E empató, P perdió (el más reciente a la derecha)
const FN = { G: 'Ganó', E: 'Empató', P: 'Perdió' };
const dots = f => (f && f.length) ? `<div class="fm" aria-label="Últimos ${f.length} partidos">${f.map(x => `<i class="${x}" title="${FN[x]}"></i>`).join('')}</div>` : '';

function team(n, logo, loc, f) {
  const img = logo ? `<img src="${logo}" onerror="this.parentNode.textContent='${ini(n).replace(/'/g, '')}'">` : ini(n);
  return `<div class="eq"><div class="ci">${img}</div><b>${loc ? '🏠 ' : ''}${n}</b>${dots(f)}</div>`;
}

// desplegable
function tog(ci) {
  const el = document.getElementById('m' + ci);
  const abierto = el.classList.toggle('open');
  el.querySelector('.mbtn').setAttribute('aria-expanded', abierto);
  abierto ? OPEN.add(CK[ci]) : OPEN.delete(CK[ci]);
}

function card(g, hist) {
  const mk = g.date + '|' + g.home + '|' + g.away;
  const ci = CK.push(mk) - 1;
  const abierto = OPEN.has(mk);
  const mid = hist
    ? `<div class="scw"><div class="pj p">Proyección ${g.proy[0]} - ${g.proy[1]}</div><div class="sc">${g.real[0]} - ${g.real[1]}</div><div class="pj">Resultado real</div></div>`
    : `<div class="sc">${g.proy[0]} - ${g.proy[1]}</div>`;
  const badge = hist ? `<span class="badge ${g.atino === 'SI' ? '' : 'incorrecto'}">${g.atino === 'SI' ? 'ACERTÓ' : 'FALLÓ'}</span>` : '';

  let pr = '', edgeH = '', valB = '';
  if (g.probs && !hist) {
    const gid = PR.push(g.probs) - 1;
    PRK[gid] = mk;
    const cols = g.probs.length === 2 ? ['#d6264f', '#6b8cff'] : ['#d6264f', '#555570', '#6b8cff'];
    pr = `<div class="pr">${g.probs.map((x, k) => `<div style="width:${(x.p * 100).toFixed(1)}%;background:${cols[k]}"></div>`).join('')}</div>
          <div class="pl">${g.probs.map(x => `<span>${x.l} ${(x.p * 100).toFixed(0)}%</span>`).join('')}</div>`;
    valB = `<span class="val" id="b${gid}"></span>`;
    edgeH = `<div class="edge"><h4><span>EDGE · RESULTADO</span><button class="fmt2" onclick="toggleFmt()">${FMT === 'us' ? 'Americano' : 'Decimal'} ⇄</button></h4>${g.probs.map((x, k) => {
      const key = mk + '|' + k;
      const d = key in ODDS ? ODDS[key] : x.c;
      const pj = Math.min(0.995, Math.max(0.005, x.p));
      return `<div class="eo"><span>${x.l}</span><span class="fj">${(x.p * 100).toFixed(0)}% · justo ${fmtO(1 / pj)}</span>
        <input type="text" inputmode="decimal" placeholder="momio" value="${fmtO(d)}" oninput="setOdd(${gid},${k},this.value)">
        <span class="ed" id="e${gid}_${k}"></span></div>`;
    }).join('')}</div>`;
  }
  const head = hist ? '<tr><td>Métrica</td><td>Proyectado</td><td>Real</td></tr>' : '<tr><td></td><td>Modelo</td><td>Tu línea</td><td>Prob · momio justo</td></tr>';
  const ctx = hist ? '' : (g.ia ? `<div class="ctx">${g.ia}</div>` : '');
  const etq = hist ? 'Proyección vs. real' : 'Momios y mercados';

  return `<div class="card">
    <div class="top"><span class="lg">${g.lg}</span><span>${valB}${badge}</span></div>
    <div class="fecha">${g.date}${g.week ? ` · SEMANA ${g.week}` : ''}${g.div ? '<span class="tag">DIVISIONAL</span>' : ''}</div>
    <div class="vs">${team(g.home, g.logo_h, true, g.form && g.form[0])}${mid}${team(g.away, g.logo_a, false, g.form && g.form[1])}</div>
    ${pr}
    <div class="mx ${abierto ? 'open' : ''}" id="m${ci}">
      <button class="mbtn" aria-expanded="${abierto}" onclick="tog(${ci})"><span>${etq}</span><span class="ar">▾</span></button>
      <div class="mbody">${edgeH}<div class="tw"><table>${head}${g.rows.map(r => rowH(r, hist, mk)).join('')}</table></div></div>
    </div>
    ${ctx}
  </div>`;
}

// ── datos por pestaña ──
const esCL = x => x.lg === 'Champions League';
function getItems(t) {
  if (t === 'NFL') return [...D.nfl];
  if (t === 'FUT') return D.fut.filter(x => !esCL(x));
  if (t === 'CL') return D.fut.filter(esCL);
  return [...D.nflhist.map(x => ({ ...x, lg: 'NFL' })), ...D.futhist.map(x => ({ ...x, lg: x.liga }))];
}
const byDate = (a, b) => new Date(a.date) - new Date(b.date);

// ── ticker de resultados ──
function ticker() {
  const el = document.getElementById('tick');
  const rs = getItems('RES').sort((a, b) => new Date(b.date) - new Date(a.date)).slice(0, 30);
  if (!rs.length) { el.style.display = 'none'; return; }
  const im = (u, n) => u ? `<img src="${u}" alt="" onerror="this.style.display='none'">` : '';
  const html = rs.map(g => `<span class="ti"><span class="lg2">${esc(g.lg)}</span>${im(g.logo_h)}${esc(g.home)} <em>${g.real[0]}</em>–<em>${g.real[1]}</em> ${esc(g.away)}${im(g.logo_a)}<span class="${g.atino === 'SI' ? 'ok' : 'no'}">${g.atino === 'SI' ? '✓' : '✗'}</span></span>`).join('');
  el.innerHTML = `<div class="lbl">RESULTADOS</div><div class="trk"><div class="tk" style="--d:${Math.max(30, rs.length * 4)}s">${html}${html}</div></div>`;
}

// ── lobby ──
const TILES = [
  ['NFL', '🏈', 'NFL'], ['FUT', '⚽', 'Fútbol'], ['CL', '🏆', 'Champions'], ['RES', '📊', 'Resultados']
];
function lobby() {
  document.getElementById('home').innerHTML = TILES.map(([k, ic, nm]) => {
    const it = getItems(k).sort(byDate);
    const logos = [...new Set(it.flatMap(g => [g.logo_h, g.logo_a]).filter(Boolean))].slice(0, 4);
    let p, nx = '';
    if (k === 'RES') {
      const ok = it.filter(g => g.atino === 'SI').length;
      p = it.length ? `${ok} de ${it.length} aciertos del modelo` : 'Sin resultados recientes';
    } else {
      p = `${it.length} partido${it.length === 1 ? '' : 's'} próximos`;
      if (it[0]) nx = `Próximo: ${esc(it[0].home)} vs ${esc(it[0].away)} · ${it[0].date}`;
    }
    return `<div class="tile t-${k}" onclick="setTab('${k}')">
      <div class="tbg">${logos.map(u => `<img src="${u}" alt="" onerror="this.style.display='none'">`).join('')}</div>
      <div class="tic">${ic}</div><h2>${nm}</h2><p>${p}</p><div class="nx">${nx}</div><span class="go">ENTRAR →</span></div>`;
  }).join('');
}

// ── navegación (con hash, para que el botón "atrás" de Safari funcione) ──
function setTab(t) {
  const h = t.toLowerCase();
  if (location.hash === '#' + h) { goTab(t); } else { location.hash = h; }
}
function goTab(t) { tab = t; lg = 'Todas'; render(); window.scrollTo(0, 0); }
window.addEventListener('hashchange', () => goTab(HASH[(location.hash || '').slice(1)] || 'HOME'));
function setLg(l) { lg = l; render(); }

function render() {
  ROWS.length = 0; ROWK.length = 0; PR.length = 0; PRK.length = 0; CK.length = 0;
  ['HOME', 'NFL', 'FUT', 'CL', 'RES'].forEach(k => document.getElementById('t' + k).classList.toggle('on', k === tab));
  document.getElementById('fmtbtn').textContent = (FMT === 'us' ? 'Americano' : 'Decimal') + ' ⇄';

  const home = tab === 'HOME', hist = tab === 'RES';
  document.getElementById('home').style.display = home ? '' : 'none';
  document.getElementById('chips').style.display = home ? 'none' : '';
  document.getElementById('grid').style.display = home ? 'none' : '';
  document.querySelector('.nota').style.display = home ? 'none' : '';
  document.getElementById('ttl').innerHTML = home ? 'Edge <i>///</i> Lab' : `${TT[tab]} <i>///</i>`;
  document.getElementById('descr').textContent = home ? 'Modelo vs. mercado · elige una sección' : SUB[tab];

  if (home) {
    lobby();
    document.getElementById('ai').innerHTML = ''; document.getElementById('picks').innerHTML = '';
    return;
  }

  let items = getItems(tab).sort(byDate);
  const ligas = (tab === 'FUT' || tab === 'RES') ? ['Todas', ...new Set(items.map(x => x.lg))] : [];
  document.getElementById('chips').innerHTML = ligas.map(l => `<div class="chip ${l === lg ? 'on' : ''}" onclick="setLg('${l}')">${l}</div>`).join('');
  if (lg !== 'Todas' && ligas.length) items = items.filter(x => x.lg === lg);

  const a = D.ai[tab];
  document.getElementById('ai').innerHTML = (a && !hist) ? `<div class="ai"><h3>RESUMEN IA</h3>${a}</div>` : '';
  const pk = (D.picks && D.picks[tab]) || [];
  document.getElementById('picks').innerHTML = (pk.length && !hist) ? `<div class="pks"><h3>PICKS RECOMENDADOS POR GEMINI</h3>${pk.map(p =>
    `<div class="pk"><span class="cf ${esc(p.confianza)}">${esc(String(p.confianza || '').toUpperCase())}</span>
     <div><b>${esc(p.eleccion)}</b>${p.cuota ? ' @' + fmtO(parseFloat(p.cuota)) : ''}
     <small>${esc(p.partido)} · ${esc(p.mercado)}</small>${esc(p.razon)}</div></div>`).join('')}</div>` : '';

  let vacio = '<div class="sub">Sin partidos en este rango.</div>';
  if ((tab === 'FUT' || tab === 'CL') && D.diag && D.diag.length) vacio += `<div class="diag"><b>Diagnóstico:</b>\n${D.diag.join('\n')}</div>`;
  document.getElementById('grid').innerHTML = items.length ? items.map(g => card(g, hist)).join('') : vacio;
  PR.forEach((_, i) => edge(i));
}

ticker();
render();
</script></body></html>"""


def descargar(ruta):
    """En Colab descarga el archivo al navegador; fuera de Colab solo avisa dónde quedó."""
    try:
        from google.colab import files
        files.download(ruta)
    except Exception:
        print(f"Archivo guardado en {os.path.abspath(ruta)} (ábrelo con tu navegador)")


def main():
    nfl_p, nfl_h = nfl()
    fut_p, fut_h = futbol()
    es_cl = lambda g: g["lg"] == "Champions League"
    grupos = (("NFL", "NFL", nfl_p),
              ("FUT", "fútbol (La Liga y Premier League)", [g for g in fut_p if not es_cl(g)]),
              ("CL", "Champions League", [g for g in fut_p if es_cl(g)]))
    resumen, picks = {}, {}
    for clave, dep, lst in grupos:
        fx = [dict(id=f"{g['home']}-{g['away']}", date=g["date"], home=g["home"], away=g["away"],
                   semana=g.get("week"), divisional=g.get("div"), proy=g["proy"],
                   probs=[dict(l=x["l"], p=round(x["p"], 3), cuota=x["c"]) for x in g["probs"]],
                   mercados=[dict(m=r["l"], modelo=r["m"], linea=r["line"], prob_si=r["p"]) for r in g["rows"]])
              for g in lst]
        nom = {f["id"]: f"{f['home']} vs {f['away']}" for f in fx}
        aj, res, pk = ia(fx, dep)
        resumen[clave] = res
        for p in pk:
            p["partido"] = nom.get(p.get("id"), p.get("id"))
        picks[clave] = pk
        for g in lst:
            g["ia"] = aj.get(f"{g['home']}-{g['away']}", {}).get("notas", "")
    data = dict(nfl=nfl_p, nflhist=nfl_h, fut=fut_p, futhist=fut_h, ai=resumen, picks=picks, diag=DIAG)
    # "</" escapado: evita que un texto de la IA con </script> rompa la página
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = HTML_PART1 + HTML_PART2_JS.replace("__DATA__", payload)
    with open(SALIDA, "w", encoding="utf-8") as f:
        f.write(html)
    print("Listo →", SALIDA)
    descargar(SALIDA)


if __name__ == "__main__":
    main()
