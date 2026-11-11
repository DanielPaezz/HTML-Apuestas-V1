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
N_FORMA = 5           # Racha y estrellas ofensiva/defensiva: últimos N juegos
MODELO_IA = "gemini-2.5-flash"
SALIDA = "index.html"
TS = pd.Timestamp
DIAG = []             # mensajes de diagnóstico (se imprimen y se muestran en el HTML si una pestaña sale vacía)

# ── Lesiones ──
LES_MAX_DIAS = 9          # ESPN: un reporte que no sea de largo plazo (IR/PUP/susp.) con más de N días se ignora
FORZAR_ACTIVO = set()     # nombres que SIEMPRE se tratan como sanos (parche manual). Vacío = no se fuerza a nadie.

# ── Modelo de QB (valor del QB titular vs. su suplente) ──
# Modo QB_OUT_ONLY: solo se aplica la penalización si el reporte lo marca Out/Inactive.
# Si el QB está activo, Questionable/Doubtful o no aparece, se usa el Ridge sin ajuste.
# Si ya calibraste con calibrar_qb.py, pega aquí los valores que te recomendó.
QB_DB_JUEGO = 36.0        # dropbacks (pases + sacks) por partido
QB_PRIOR_DB = 150.0       # fuerza del "jalón" hacia el promedio (más alto = más conservador)
QB_HL = 150               # vida media (días) del peso de cada partido, igual que el Ridge
QB_ESCALA = 1.0           # perilla global: 0.8 = penalizaciones 20 % más chicas
QB_MAX_PTS = 10.0         # tope de puntos que vale un QB sobre su suplente
QB_BONO_MAX = 2.0         # tope del ajuste positivo cuando el titular SÍ juega
QB_CORREGIR_JUEGA = False # modo QB_OUT_ONLY: si el titular está activo, se conserva el Ridge normal
QB_RATING = {}   # {nombre_normalizado: rating EPA/dropback} de TODOS los QB con historial, sin importar el equipo
QB_PEN_DEF = 3.5          # penalización fija si no hay datos de EPA
QB_YDS_K = 0.8            # cuánto se mueven las yardas aéreas del equipo por cada punto de ajuste de QB
PTS_EQUIPO = 22.0         # puntos promedio por equipo (para escalar yardas)


def diag(msg):
    DIAG.append(msg)
    print("·", msg)


def num(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except Exception:
        return None


def fila(l, m, d, sd=None, line=None, o="Over", u="Under", p=None, real=None):
    return dict(l=l, m=round(float(m), 2), d=d, sd=sd, line=line, o=o, u=u, p=p, real=real)


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def ml_dec(x):
    """Moneyline americano -> cuota decimal (internamente todo se guarda en decimal)."""
    x = num(x)
    if x is None or x == 0:
        return None
    return round(1 + x / 100, 3) if x > 0 else round(1 + 100 / abs(x), 3)


# ───────────────────────── Racha y estrellas (últimos N juegos) ─────────────────────────
def racha_equipo(gl, t, ref, n=N_FORMA):
    """Resultado de los últimos n juegos del equipo, del más viejo al más reciente: G (ganó), E (empató), P (perdió).
    gl: DataFrame con Date, h, a, gh, ga."""
    g = gl[(gl.Date < ref) & ((gl.h == t) | (gl.a == t))].sort_values("Date").tail(n)
    out = ""
    for r in g.itertuples():
        mio, rival = (r.gh, r.ga) if r.h == t else (r.ga, r.gh)
        out += "G" if mio > rival else ("P" if mio < rival else "E")
    return out


def estrellas_liga(gl, ref, n=N_FORMA, minj=1):
    """Estrellas 1-5 de ofensiva y defensiva (5 = mejor) con los últimos n juegos de CADA equipo de la liga.
    1) Cada juego se ajusta por el rival: los puntos/goles a favor se descuentan si el rival suele conceder mucho,
       y los concedidos se descuentan si el rival suele anotar mucho (promedios de todos los equipos antes de ref).
    2) Se promedian los últimos n juegos ajustados de cada equipo.
    3) Se ordenan TODOS los equipos y se reparten en quintiles: el 20 % mejor = 5 estrellas ... el 20 % peor = 1.
    Devuelve {equipo: dict(of, de, gf, gc, n)} con gf/gc sin ajustar (para mostrar)."""
    gl = gl[gl.Date < ref]
    if gl.empty:
        return {}
    largo = pd.concat([
        pd.DataFrame(dict(t=gl.h, opp=gl.a, Date=gl.Date, gf=gl.gh, gc=gl.ga)),
        pd.DataFrame(dict(t=gl.a, opp=gl.h, Date=gl.Date, gf=gl.ga, gc=gl.gh))]).sort_values("Date")
    prom = largo.groupby("t")[["gf", "gc"]].mean()          # lo que cada equipo suele anotar / conceder
    liga = float(largo.gf.mean())                           # promedio de la liga
    largo["gfa"] = largo.gf - (largo.opp.map(prom.gc) - liga).fillna(0)
    largo["gca"] = largo.gc - (largo.opp.map(prom.gf) - liga).fillna(0)
    res = {}
    for t, g in largo.groupby("t"):
        u = g.tail(n)
        if len(u) >= minj:
            res[t] = dict(gf=float(u.gf.mean()), gc=float(u.gc.mean()),
                          gfa=float(u.gfa.mean()), gca=float(u.gca.mean()), n=int(len(u)))
    if len(res) < 4:
        return {}

    def rank(key, clave, mayor_mejor):
        orden = sorted(res, key=lambda t: res[t][key], reverse=mayor_mejor)
        N = len(orden)
        for i, t in enumerate(orden):
            res[t][clave] = 5 - min(4, int(i / N * 5))

    rank("gfa", "of", True)
    rank("gca", "de", False)
    return res


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
                  "mercado la media del modelo y la línea de referencia. El campo 'qb_ajuste' (si existe) indica si el QB "
                  "titular está fuera o dudoso y cuántos puntos ya restó el modelo.\n"
                  f"{json.dumps(fx, ensure_ascii=False)}\n"
                  "Busca en la web lesiones, alineaciones probables, clima y noticias recientes de esos partidos y "
                  "combínalo con el modelo. Devuelve SOLO un objeto JSON (sin markdown) con esta forma: " + esq + ". "
                  "Reglas: máximo 6 picks; solo si hay valor real (prob. del modelo × cuota > 1, o media del modelo "
                  "claramente por encima/debajo de la línea) y las noticias no lo contradicen; si no hay valor, picks vacío. "
                  "Nunca propongas un pick sobre un jugador lesionado, dudoso o que no sea titular (revisa 'bajas'). "
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
MIN_VOL = {"QB": 15, "RB": 6, "WR": 3}   # un juego solo cuenta para la proyección si tuvo al menos N intentos / acarreos / targets
SD_MARGEN = 13.5

DIVS = {"AFC Este": "BUF MIA NE NYJ", "AFC Norte": "BAL CIN CLE PIT", "AFC Sur": "HOU IND JAX TEN",
        "AFC Oeste": "DEN KC LV LAC", "NFC Este": "DAL NYG PHI WAS", "NFC Norte": "CHI DET GB MIN",
        "NFC Sur": "ATL CAR NO TB", "NFC Oeste": "ARI LA SF SEA"}
DIV = {t: d for d, ts in DIVS.items() for t in ts.split()}

# Apodo (última palabra del nombre en ESPN) -> abreviatura de nflverse
NFL_NICK = {"Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF", "Panthers": "CAR", "Bears": "CHI",
            "Bengals": "CIN", "Browns": "CLE", "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
            "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC", "Raiders": "LV", "Chargers": "LAC",
            "Rams": "LA", "Dolphins": "MIA", "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
            "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF", "Seahawks": "SEA", "Buccaneers": "TB",
            "Titans": "TEN", "Commanders": "WAS"}


def es_div(r, h, a):
    v = getattr(r, "div_game", None)          # columna de nflverse; si no existe, se usa el diccionario
    if v is not None and v == v:
        return bool(int(v))
    return DIV.get(h) is not None and DIV.get(h) == DIV.get(a)


def _nj(s):
    """Nombre de jugador normalizado y sin sufijos (Jr., II...), para cruzar ESPN con nflverse."""
    t = _n(s).split()
    while len(t) > 1 and t[-1] in ("jr", "sr", "ii", "iii", "iv", "v"):
        t.pop()
    return " ".join(t)


SANO = ("", "probable", "nan", "none", "active", "activo")
LARGA = ("reserve", "pup", "suspen", "nfi", "non-football", "physically unable")


def _ts(x):
    """Cualquier fecha -> Timestamp sin zona horaria (o None)."""
    try:
        t = pd.to_datetime(x, utc=True, errors="coerce")
        return None if pd.isna(t) else t.tz_convert(None)
    except Exception:
        return None


def lesionados(nfr, temp, semanas):
    """{semana: {equipo: {nombre_normalizado: dict(n, pos, est, src)}}}.
    Fuentes: ESPN (reporte vigente) + nflverse (reporte de CADA semana). Entre las dos gana la más reciente.
    Un reporte de ESPN que no sea de largo plazo (IR/PUP/susp.) y tenga más de LES_MAX_DIAS se descarta.
    Cualquier estado distinto de 'Probable' (Out, Doubtful, Questionable, Day-To-Day, IR...) cuenta como lesión."""
    ahora = pd.Timestamp.now(tz="UTC").tz_convert(None)
    forzar = {_nj(x) for x in FORZAR_ACTIVO}
    viejos, espn_c = [], {}

    def add(d, t, nombre, pos, est, f, src):
        if not t or not nombre:
            return
        d.setdefault((t, _nj(nombre)), []).append(
            dict(n=str(nombre), pos=str(pos or ""), est=str(est or "").strip(), f=f, src=src))

    try:
        r = requests.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries", timeout=30)
        r.raise_for_status()
        for eq in r.json().get("injuries", []):
            t = NFL_NICK.get(str(eq.get("displayName", "")).split(" ")[-1])
            for x in eq.get("injuries", []):
                a = x.get("athlete") or {}
                est = str(x.get("status") or "").strip()
                f = _ts(x.get("date"))
                if f is not None and not any(k in est.lower() for k in LARGA) and (ahora - f).days > LES_MAX_DIAS:
                    viejos.append(f"{a.get('displayName')} ({est}, {f:%d-%b})")
                    continue
                add(espn_c, t, a.get("displayName"), (a.get("position") or {}).get("abbreviation"), est, f, "ESPN")
    except Exception as e:
        diag(f"NFL lesiones: ESPN falló ({e})")
    if viejos:
        diag(f"NFL lesiones: ESPN con reporte viejo ignorado: {viejos[:8]}")

    try:
        nv = nfr.load_injuries(temp).to_pandas()
    except Exception as e:
        nv = None
        diag(f"NFL lesiones: nflverse no disponible ({e})")

    out = {}
    for s in sorted(set(semanas)):
        c = {k: list(v) for k, v in espn_c.items()}
        if nv is not None:
            for r in nv[nv.week == s].itertuples():
                add(c, NFL_TEAM.get(r.team, r.team), r.full_name, r.position, r.report_status,
                    _ts(getattr(r, "date_modified", None)), "nflverse")
        inj = {}
        for (t, nn), lst in c.items():
            lst.sort(key=lambda z: (z["f"] or pd.Timestamp.min, z["src"] == "nflverse"))
            z = lst[-1]                                   # el reporte más reciente manda
            if z["est"].lower() in SANO or nn in forzar:
                continue
            inj.setdefault(t, {})[nn] = dict(n=z["n"], pos=z["pos"], est=z["est"], src=z["src"])
        out[s] = inj
        diag(f"NFL lesiones semana {s}: {sum(len(v) for v in inj.values())} jugadores en {len(inj)} equipos")
    return out


def es_fuera(est):
    """True solo para estados que confirman que el jugador no estará disponible.
    Questionable/Doubtful/Day-To-Day NO activan el ajuste; se conserva el Ridge normal."""
    e = str(est or "").strip().lower()
    return e in ("out", "inactive", "inactivo", "not active", "not playing") or any(
        k in e for k in ("reserve", "pup", "suspen", "nfi", "non-football", "physically unable")
    )


def partir_lesiones(inj):
    """bad: quienes NO jugarán. duda: quienes están dudosos (Questionable, Doubtful, Day-To-Day)."""
    bad = {t: frozenset(n for n, x in v.items() if es_fuera(x["est"])) for t, v in inj.items()}
    duda = {t: {n: x["est"] for n, x in v.items() if not es_fuera(x["est"])} for t, v in inj.items()}
    return bad, duda


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
            pc = ps[ps.Date < ref].copy()
            pc["nj"] = pc.player_display_name.map(_nj)          # nombre normalizado para cruzar con el depth chart
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


def ranking(M, team, pos, ref, activo=True):
    """Jugadores del equipo en la posición, de mayor a menor volumen reciente (intentos / acarreos / targets).
    activo=True: solo quienes jugaron en el ÚLTIMO partido del equipo (proxy de titular sano y vigente)."""
    vcol = POS[pos][1]
    d = M["ps"]
    de = d[d.team == team]
    if de.empty:
        return []
    ult = de.Date.max()
    d = d[d.position == pos]
    rec = d[(d.team == team) & (d.Date >= ref - dt.timedelta(days=42))]
    if rec.empty:
        rec = d[d.team == team]
    if rec.empty:
        return []
    g = rec.groupby(["player_id", "player_display_name"])[vcol].sum().sort_values(ascending=False)
    g = g[g > 0]
    if activo:
        jugo = set(d.loc[(d.team == team) & (d.Date == ult), "player_id"])
        g = g[[pid in jugo for pid, _ in g.index]]
    return list(g.index)


# ── Depth chart de ESPN (QB / RB / WR) ──
DEPTH = {}   # {equipo: {"QB": [nombres en orden], "RB": [...], "WR": [...]}}


def _atleta(a_):
    """(nombre, id) de un atleta del depth chart; ESPN a veces solo manda el id o un $ref."""
    ath = a_.get("athlete") if isinstance(a_.get("athlete"), dict) else a_
    nombre = ath.get("displayName") or ath.get("fullName") or a_.get("displayName") or a_.get("fullName")
    pid = ath.get("id") or a_.get("id")
    if not pid:
        ref = ath.get("$ref") or a_.get("$ref") or ""
        m = re.search(r"athletes/(\d+)", str(ref))
        pid = m.group(1) if m else None
    return nombre, (str(pid) if pid else None)


def _parse_depth(j):
    """Recorre el JSON del depth chart y junta [(nombre, id)] de QB/RB/WR en el orden de ESPN (qb, rb, wr1, wr2...)."""
    out = {"QB": [], "RB": [], "WR": []}

    def walk(o):
        if isinstance(o, dict):
            pos = o.get("positions")
            if isinstance(pos, dict):
                for k, v in sorted(pos.items(), key=lambda kv: str(kv[0])):
                    p = str(k).upper().rstrip("0123456789")
                    if p in out and isinstance(v, dict):
                        for a_ in v.get("athletes", []) or []:
                            if not isinstance(a_, dict):
                                continue
                            n, i = _atleta(a_)
                            if (n or i) and not any((n and n == x[0]) or (i and i == x[1]) for x in out[p]):
                                out[p].append((n, i))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(j)
    return out


def _roster_ids(tid):
    """{id: nombre} de todo el roster de ESPN del equipo (para resolver atletas que vienen solo con id)."""
    try:
        j = _json(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{tid}/roster")
    except Exception:
        return {}
    ath = []
    for g in j.get("athletes", []):
        ath += g.get("items", []) if isinstance(g, dict) and "items" in g else [g]
    return {str(a_.get("id")): (a_.get("fullName") or a_.get("displayName"))
            for a_ in ath if isinstance(a_, dict) and a_.get("id")}


def nfl_depth():
    """Depth chart oficial de ESPN por equipo. Llena DEPTH con nombres, en el orden de ESPN."""
    try:
        d = _json("https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams")
        eqs = [t.get("team", {}) for sp in d.get("sports", []) for lg in sp.get("leagues", []) for t in lg.get("teams", [])]
    except Exception as e:
        diag(f"Depth chart: no se pudo leer la lista de equipos ({e})")
        return

    def uno(t):
        try:
            j = _json(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{t['id']}/depthcharts")
            dc = _parse_depth(j)
        except Exception:
            return t, None
        if any(n is None for lst in dc.values() for n, _ in lst):      # faltan nombres: se resuelven por id
            ids = _roster_ids(t["id"])
            dc = {p: [(n or ids.get(i), i) for n, i in lst] for p, lst in dc.items()}
        return t, {p: [n for n, _ in lst if n] for p, lst in dc.items()}

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(uno, [t for t in eqs if t.get("id")]))
    DEPTH.clear()
    for t, dc in res:
        ab = str(t.get("abbreviation", "")).upper()
        ab = NFL_TEAM.get(ab, ab)
        if dc and any(dc.values()):
            DEPTH[ab] = dc
    diag(f"Depth chart ESPN: {len(DEPTH)} equipos"
         + ("" if DEPTH else " (vacío: se usa el volumen reciente como respaldo)"))


def _proy_jugador(d, pid, nombre, ycol, pos, M, dfn_opp):
    h = d[d.player_id == pid]
    hv = h[h[POS[pos][1]] >= MIN_VOL[pos]]        # descarta entradas de relevo / garbage time
    if len(hv):
        h = hv
    h = h.sort_values("Date", ascending=False).head(8)
    n = len(h)
    if n == 0:
        return None
    w = 0.7 ** np.arange(n)
    mu = float(np.average(h[ycol], weights=w)) * dfn_opp
    var_j = float(h[ycol].var(ddof=0)) if n > 1 else 0.0
    sd = math.sqrt((n * var_j + 4 * M["sd_pos"][pos] ** 2) / (n + 4))
    return pid, nombre, mu, max(sd, 10.0)


def jugador(M, team, pos, ref, dfn_opp, bad=frozenset(), activo=True, depth=True):
    """Titular de la posición que se usará para la apuesta. Primero el depth chart de ESPN (en orden,
    saltando a quien esté en `bad`); si ESPN no lo trae o nadie tiene estadísticas, usa el volumen reciente.
    depth=False (resultados pasados): no se usa el depth chart actual, solo el volumen de esa fecha."""
    ycol = POS[pos][0]
    d = M["ps"]
    d = d[d.position == pos]
    if depth and DEPTH.get(team, {}).get(pos):
        y, v, thr = POS[pos]
        for nombre in DEPTH[team][pos]:
            k = _nj(nombre)
            if k in bad:
                continue
            sub = d[(d.team == team) & (d.nj == k)]
            if sub.empty:
                sub = d[d.nj == k]                    # estadísticas con otro equipo (fichaje reciente)
            r = None
            if not sub.empty:
                sub = sub.sort_values("Date")
                r = _proy_jugador(d, sub.player_id.iloc[-1], sub.player_display_name.iloc[-1], y, pos, M, dfn_opp)
            if not r:
                # sin historial (novato / de práctica): se respeta el depth chart con un promedio de la posición
                base = d.loc[d[v] >= thr, y]
                mu = float(base.median()) * dfn_opp if len(base) else 0.0
                r = (None, nombre, mu, max(M["sd_pos"][pos], 10.0))
            return r
        # todos los del depth chart están fuera: cae al respaldo
    for pid, nombre in ranking(M, team, pos, ref, activo=activo):      # respaldo
        if _nj(nombre) in bad:
            continue
        r = _proy_jugador(d, pid, nombre, ycol, pos, M, dfn_opp)
        if r:
            return r
    return None


def titulares(M, team, pos, ref, k):
    """Los k primeros del depth chart (o del ranking por volumen si ESPN no trae datos)."""
    dep = DEPTH.get(team, {}).get(pos, [])[:k]
    return dep if dep else [n for _, n in ranking(M, team, pos, ref, activo=False)[:k]]


def bajas_clave(M, team, ref, inj):
    """QB1, RB1, WR1 y WR2 según el depth chart de ESPN que estén en el reporte de lesiones."""
    if "ps" not in M:
        return []
    rep = inj.get(team, {})
    out = []
    for pos, k in (("QB", 1), ("RB", 1), ("WR", 2)):
        for nombre in titulares(M, team, pos, ref, k):
            x = rep.get(_nj(nombre))
            if x:
                out.append(dict(n=nombre, pos=pos, est=x["est"]))
    return out


# ── Salario NFL (OverTheCap vía nflverse) para elegir QB reemplazo (respaldo si no hay depth chart) ──
# Se usa el salario base de la temporada actual (millones USD); si no existe,
# se utiliza APY como respaldo. No es salario fantasy ni valor de subasta.
TEAM_CONTRACT = {
    "arizona": "ARI", "cardinals": "ARI", "atlanta": "ATL", "falcons": "ATL",
    "baltimore": "BAL", "ravens": "BAL", "buffalo": "BUF", "bills": "BUF",
    "carolina": "CAR", "panthers": "CAR", "chicago": "CHI", "bears": "CHI",
    "cincinnati": "CIN", "bengals": "CIN", "cleveland": "CLE", "browns": "CLE",
    "dallas": "DAL", "cowboys": "DAL", "denver": "DEN", "broncos": "DEN",
    "detroit": "DET", "lions": "DET", "green bay": "GB", "packers": "GB",
    "houston": "HOU", "texans": "HOU", "indianapolis": "IND", "colts": "IND",
    "jacksonville": "JAX", "jaguars": "JAX", "kansas city": "KC", "chiefs": "KC",
    "las vegas": "LV", "raiders": "LV", "los angeles rams": "LA", "rams": "LA",
    "los angeles chargers": "LAC", "chargers": "LAC", "miami": "MIA", "dolphins": "MIA",
    "minnesota": "MIN", "vikings": "MIN", "new england": "NE", "patriots": "NE",
    "new orleans": "NO", "saints": "NO", "new york giants": "NYG", "giants": "NYG",
    "new york jets": "NYJ", "jets": "NYJ", "philadelphia": "PHI", "eagles": "PHI",
    "pittsburgh": "PIT", "steelers": "PIT", "san francisco": "SF", "49ers": "SF",
    "seattle": "SEA", "seahawks": "SEA", "tampa bay": "TB", "buccaneers": "TB",
    "tennessee": "TEN", "titans": "TEN", "washington": "WAS", "commanders": "WAS"
}

ROSTER_QB = {}   # {equipo: [nombres de QB activos]}
SAL_NOM = {}     # {nombre_normalizado: salario en millones}


def nfl_qb_roster(nfr, temp):
    """QB activos por equipo (roster nflverse). Sirve para no elegir QB que ya no están en el equipo."""
    try:
        df = nfr.load_rosters(temp).to_pandas()
        df = df[df.position.astype(str).str.upper().eq("QB")]
        if "status" in df.columns:
            df = df[df.status.astype(str).str.upper().eq("ACT")]
        col = "full_name" if "full_name" in df.columns else "player_name"
        out = {}
        for r in df.itertuples():
            t = NFL_TEAM.get(r.team, r.team)
            out.setdefault(t, []).append(str(getattr(r, col)))
        ROSTER_QB.clear()
        ROSTER_QB.update(out)
        diag(f"Roster NFL: {sum(map(len, out.values()))} QB activos en {len(out)} equipos")
    except Exception as e:
        diag(f"Roster NFL no disponible ({type(e).__name__}: {e})")


ESPN_SAL = {}    # {nombre_normalizado: salario en millones} según ESPN (tiene prioridad sobre nflverse)


def _buscar_salario(o, prof=0):
    """Busca recursivamente un salario (>0) en el JSON de ESPN. Devuelve millones de USD o None."""
    if prof > 6:
        return None
    if isinstance(o, dict):
        for k in ("salary", "baseSalary", "base_salary", "capHit", "cap_hit", "totalCash"):
            v = num(o.get(k))
            if v is not None and v > 0:
                return v / 1e6 if v > 1000 else v
        for v in o.values():
            r = _buscar_salario(v, prof + 1)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _buscar_salario(v, prof + 1)
            if r:
                return r
    return None


def _json(url, **prm):
    r = requests.get(url, params=prm or None, timeout=25)
    r.raise_for_status()
    return r.json()


def _salario_core(temp, pid):
    """Respaldo: contrato del jugador en la API 'core' de ESPN."""
    base = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{temp}/athletes/{pid}/contracts"
    try:
        d = _json(base, lang="en", region="us")
        r = _buscar_salario(d)
        if r:
            return r
        for it in d.get("items", []):
            if isinstance(it, dict) and it.get("$ref"):
                try:
                    r = _buscar_salario(_json(it["$ref"].replace("http://", "https://")))
                    if r:
                        return r
                except Exception:
                    pass
    except Exception:
        pass
    return None


def nfl_qb_espn(temp):
    """PRIORIDAD ESPN: roster de QB por equipo (con su estatus) y salario. Sobrescribe ROSTER_QB y ESPN_SAL.
    Salario: primero el que venga en el roster de ESPN; si no, el contrato de la API core de ESPN."""
    try:
        d = _json("https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams")
        eqs = [t.get("team", {}) for sp in d.get("sports", []) for lg in sp.get("leagues", []) for t in lg.get("teams", [])]
    except Exception as e:
        diag(f"ESPN QB: no se pudo leer la lista de equipos ({e})")
        return

    def uno(t):
        try:
            j = _json(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{t['id']}/roster")
        except Exception:
            return t, []
        ath = []
        for g in j.get("athletes", []):
            ath += g.get("items", []) if isinstance(g, dict) and "items" in g else [g]
        out = []
        for a_ in ath:
            if not isinstance(a_, dict):
                continue
            if str((a_.get("position") or {}).get("abbreviation", "")).upper() != "QB":
                continue
            st = a_.get("status") or {}
            est = str(st.get("name") or st.get("type") or "")
            out.append(dict(id=a_.get("id"), n=a_.get("fullName") or a_.get("displayName"),
                            est=est, sal=_buscar_salario(a_.get("contract")) or _buscar_salario(a_)))
        return t, out

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(uno, [t for t in eqs if t.get("id")]))

    falta = [(t, q) for t, qs in res for q in qs if q["sal"] is None and q["id"]]
    if falta:
        with ThreadPoolExecutor(8) as ex:
            for (t, q), sal in zip(falta, ex.map(lambda tq: _salario_core(temp, tq[1]["id"]), falta)):
                q["sal"] = sal

    n_eq = n_sal = 0
    for t, qs in res:
        ab = NFL_TEAM.get(str(t.get("abbreviation", "")).upper(), str(t.get("abbreviation", "")).upper())
        qs = [q for q in qs if q["n"] and not es_fuera(q["est"])]
        if not qs:
            continue
        ROSTER_QB[ab] = [q["n"] for q in qs]
        n_eq += 1
        for q in qs:
            if q["sal"]:
                ESPN_SAL[_nj(q["n"])] = float(q["sal"])
                n_sal += 1
    diag(f"ESPN QB: {n_eq} equipos con roster, {n_sal} QB con salario")


def nfl_qb_salarios(nfr):
    """Devuelve {equipo: [{n, salario}]} desde contratos OverTheCap/nflverse.
    salario está expresado en millones USD por temporada.
    """
    try:
        raw = nfr.load_contracts()
        df = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        if df.empty:
            raise RuntimeError("dataset de contratos vacío")
        df = df[df.position.astype(str).str.upper().eq("QB")].copy()
        if "is_active" in df:
            df = df[df.is_active.fillna(False).astype(bool)]
        def team_code(v):
            x = _n(str(v)).lower().strip()
            if x in TEAM_CONTRACT:
                return TEAM_CONTRACT[x]
            for k, val in TEAM_CONTRACT.items():
                if k in x:
                    return val
            return None
        df["team_code"] = df.team.map(team_code)
        df = df[df.team_code.notna()].copy()
        season = HOY.year
        def current_salary(row):
            hist = row.get("season_history")
            if isinstance(hist, pd.DataFrame):
                hist = hist.to_dict("records")
            if isinstance(hist, str):
                try: hist = json.loads(hist)
                except Exception: hist = []
            if isinstance(hist, (list, tuple)):
                for item in hist:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("year", "")).strip() == str(season):
                        for key in ("base_salary", "cash_paid", "cap_number"):
                            v = num(item.get(key))
                            if v is not None and v > 0:
                                return float(v)
            v = num(row.get("apy"))
            return float(v) if v is not None and v > 0 else 0.0
        df["salario"] = df.apply(current_salary, axis=1)
        out = {}
        for _, row in df.iterrows():
            name = str(row.get("player") or "").strip()
            salary = float(row.get("salario") or 0)
            team = row.get("team_code")
            if name and team and salary > 0:
                out.setdefault(team, []).append({"n": name, "salario": salary})
        for team in out:
            # Un QB puede tener contratos activos anteriores; se conserva el mayor salario reportado.
            por_nombre = {}
            for z in out[team]:
                k = _nj(z["n"])
                if k not in por_nombre or z["salario"] > por_nombre[k]["salario"]:
                    por_nombre[k] = z
            out[team] = sorted(por_nombre.values(), key=lambda z: z["salario"], reverse=True)
        SAL_NOM.clear()
        for t_, lst_ in out.items():
            for z_ in lst_:
                k_ = _nj(z_["n"])
                SAL_NOM[k_] = max(SAL_NOM.get(k_, 0.0), z_["salario"])
        diag(f"Salarios NFL (OverTheCap/nflverse): {sum(map(len, out.values()))} QB cargados")
        return out
    except Exception as e:
        diag(f"Salarios NFL no disponibles ({type(e).__name__}: {e}); se usará uso reciente")
        return {}


def qb_espn_no_disponible(z, rep):
    """Descarta lesionados según ESPN Fantasy y el reporte ESPN/nflverse del modelo."""
    st = str(z.get("injury_status") or "").upper()
    if not z.get("activo", True) or st in ("OUT", "INJURY_RESERVE", "IR", "SUSPENDED", "PUP", "NFI"):
        return True
    x = rep.get(_nj(z.get("n", "")), {})
    return es_fuera(x.get("est", ""))


# ───────────────────────── Modelo de QB ─────────────────────────
def qb_modelo(ps, ref):
    """{equipo: dict(n, s, delta, bk, q_s, q_b)} — valor del QB titular vs. su suplente, en puntos por partido.
    1) EPA por dropback (pases + sacks) ponderado por recencia.
    2) Encogimiento: titulares hacia el promedio de titulares, suplentes hacia el nivel de reemplazo.
    3) delta = (EPA titular - EPA suplente) x dropbacks por juego, acotado a 0.5-QB_MAX_PTS.
    s = % de las jugadas recientes del equipo que tomó el titular (el Ridge ya trae esa mezcla)."""
    try:
        if ps is None or "passing_epa" not in ps.columns:
            diag(f"QB: no hay passing_epa en las stats; se usa penalización fija de {QB_PEN_DEF} pts")
            return {}
        d = ps[(ps.position == "QB") & (ps.Date < ref)].copy()
        sk = next((c for c in ("sacks_suffered", "sacks") if c in d.columns), None)
        d["db"] = d.attempts + (pd.to_numeric(d[sk], errors="coerce").fillna(0) if sk else 0)
        d["epa"] = pd.to_numeric(d.passing_epa, errors="coerce").fillna(0)
        d = d[(d.db >= 5)].dropna(subset=["Date"])            # fuera pases de trucos / entradas de 1-2 jugadas
        if len(d) < 40:
            return {}
        d["w"] = 0.5 ** ((ref - d.Date).dt.days / QB_HL)
        d["wdb"], d["wepa"] = d.w * d.db, d.w * d.epa

        por = d.groupby(["team", "player_id"])[["wdb", "wepa"]].sum().reset_index()
        top = por.loc[por.groupby("team").wdb.idxmax()]
        es_st = por.player_id.isin(set(top.player_id))
        mu_st = por[es_st].wepa.sum() / max(por[es_st].wdb.sum(), 1e-9)          # EPA/jugada típico de un titular
        bk_ = por[~es_st]
        mu_rep = bk_.wepa.sum() / bk_.wdb.sum() if bk_.wdb.sum() > 0 else mu_st - 0.15   # nivel de reemplazo

        pj = d.groupby("player_id")[["wdb", "wepa"]].sum()

        def rating(pid, mu):
            x = pj.loc[pid]
            return float((x.wepa + QB_PRIOR_DB * mu) / (x.wdb + QB_PRIOR_DB))

        nom_ = d.drop_duplicates("player_id").set_index("player_id").player_display_name
        QB_RATING.clear()
        for pid_ in pj.index:
            QB_RATING[_nj(nom_[pid_])] = rating(pid_, mu_rep)      # prior de suplente: conservador con quien casi no ha jugado

        Q = {}
        for t, g in d.groupby("team"):
            rec = g[g.Date >= ref - dt.timedelta(days=42)]
            base = rec if len(rec) else g
            pid = base.groupby("player_id").db.sum().sort_values(ascending=False).index[0]
            tot = g.wdb.sum()
            s = float(g.loc[g.player_id == pid, "wdb"].sum() / tot) if tot > 0 else 1.0
            nombre = g.loc[g.player_id == pid, "player_display_name"].iloc[0]
            q_s = rating(pid, mu_st)
            # Ordena los posibles reemplazos por uso RECIENTE, no por volumen histórico.
            # Así un QB veterano que apenas jugó una jugada no desplaza al QB3 que acaba de entrar.
            rec_g = g[g.Date >= ref - dt.timedelta(days=42)]
            base_rep = rec_g if len(rec_g) else g
            usos = base_rep.groupby("player_id").db.sum().sort_values(ascending=False)
            q_alts = []
            for bid in usos.index:
                if bid == pid:
                    continue
                bg = g[g.player_id == bid]
                if bg.empty:
                    continue
                q_alts.append(dict(pid=str(bid), n=bg.player_display_name.iloc[0],
                                   q=float(rating(bid, mu_rep)), uso=float(usos.loc[bid])))
            if q_alts:
                q_b, bn = q_alts[0]["q"], q_alts[0]["n"]
            else:
                q_b, bn = mu_rep, "suplente no identificado"
            delta = float(np.clip((q_s - q_b) * QB_DB_JUEGO * QB_ESCALA, 0.5, QB_MAX_PTS))
            Q[t] = dict(n=nombre, s=min(1.0, s), delta=delta, bk=bn, q_s=q_s, q_b=q_b,
                        q_alts=q_alts, mu_rep=float(mu_rep))
        return Q
    except Exception as e:
        diag(f"QB: modelo falló ({type(e).__name__}: {e})")
        return {}


def escenario_qb(Q, team, inj, M, ref, salarios_qb=None):
    """Estado del QB titular y su ajuste en PUNTOS del equipo en cada escenario.
    estado: ok (sano) | duda (Questionable/Doubtful/Day-To-Day) | fuera (Out/IR/...).
    juega / fuera: puntos que se suman (+) o restan (-) al equipo en cada caso.
    El reemplazo es el siguiente QB del depth chart de ESPN que no esté en el reporte de lesiones;
    si ESPN no lo trae, se usa salario / uso reciente."""
    q = Q.get(team)
    if not q and "ps" in M:                                   # respaldo: sin EPA, penalización fija
        rk = ranking(M, team, "QB", ref, activo=False)
        if rk:
            q = dict(n=rk[0][1], s=1.0, delta=QB_PEN_DEF, bk="suplente")
    if not q:
        return None
    rep = inj.get(team, {})
    x = rep.get(_nj(q["n"]))
    est = x["est"] if x else ""
    estado = "ok" if not x else ("fuera" if es_fuera(est) else "duda")

    bk = q.get("bk", "suplente no identificado")
    q_s = float(q.get("q_s", 0.0))
    q_b = float(q.get("q_b", 0.0))
    dl = float(q["delta"])

    # 1) PRIORIDAD: siguiente QB del depth chart de ESPN que no esté en el reporte de lesiones
    dep = []
    if estado != "ok":
        dep = [n_ for n_ in DEPTH.get(team, {}).get("QB", [])
               if _nj(n_) != _nj(q["n"]) and _nj(n_) not in rep]
    if dep:
        bk = dep[0]
        coincide = next((z for z in q.get("q_alts", []) if _nj(z["n"]) == _nj(bk)), None)
        if coincide:
            q_b = float(coincide["q"])
        elif _nj(bk) in QB_RATING:                        # llegó de otro equipo: se usa su EPA de allá
            q_b = float(QB_RATING[_nj(bk)])
        else:
            q_b = float(q.get("mu_rep", q_b))
        dl = float(np.clip((q_s - q_b) * QB_DB_JUEGO * QB_ESCALA, 0.5, QB_MAX_PTS))
        diag(f"QB {team}: reemplazo por depth chart ESPN → {bk} (EPA/dropback {q_b:+.3f}, "
             f"{'con historial' if (coincide or _nj(bk) in QB_RATING) else 'sin historial: nivel de reemplazo'})")

    # 2) RESPALDO (solo si el depth chart no dio a nadie): salario NFL / uso reciente
    elif estado == "fuera":
        # Candidatos = QB activos del roster (más los de contratos), SIN ninguno que aparezca en el reporte de lesiones.
        pool = {}
        for z in (salarios_qb or {}).get(team, []):
            pool[_nj(z["n"])] = [z["n"], float(z["salario"])]
        for n_ in ROSTER_QB.get(team, []):
            k_ = _nj(n_)
            pool.setdefault(k_, [n_, 0.0])
            pool[k_][1] = max(pool[k_][1], SAL_NOM.get(k_, 0.0))
        roster = {_nj(n_) for n_ in ROSTER_QB.get(team, [])}
        if roster:
            pool = {k: v for k, v in pool.items() if k in roster}
        for k_, v_ in pool.items():                       # ESPN manda sobre nflverse
            if k_ in ESPN_SAL:
                v_[1] = ESPN_SAL[k_]
        disp = sorted(((k, v) for k, v in pool.items()
                       if k != _nj(q["n"]) and k not in rep and v[1] > 0),
                      key=lambda kv: -kv[1][1])
        candidatos = q.get("q_alts", [])
        diag(f"QB {team} pool de reemplazo (roster={len(roster)} QB): " + "; ".join(
            f"{v[0]} ${v[1]:.1f}M" + (" [en reporte de lesiones]" if k in rep else "")
            + (" [titular]" if k == _nj(q["n"]) else "") for k, v in pool.items()) )
        if disp:
            bk, sal = disp[0][1]
            coincide = next((z for z in candidatos if _nj(z["n"]) == _nj(bk)), None)
            q_b = float(coincide["q"]) if coincide else float(QB_RATING.get(_nj(bk), q.get("mu_rep", q_b)))
            diag(f"QB {team}: reemplazo por salario NFL → {bk} (${sal:.2f}M/año)")
        else:
            # Respaldo: uso reciente, excluyendo a cualquiera que esté en el reporte de lesiones.
            elegido = next((z for z in candidatos if _nj(z["n"]) not in rep), None)
            if elegido:
                bk = elegido["n"]
                q_b = float(elegido["q"])
            else:
                bk = "QB disponible no identificado"
                q_b = float(q.get("mu_rep", q_b))
        # Recalcula la diferencia con el reemplazo elegido.
        dl = float(np.clip((q_s - q_b) * QB_DB_JUEGO * QB_ESCALA, 0.5, QB_MAX_PTS))

    s = float(q["s"])
    juega = min(QB_BONO_MAX, (1 - s) * dl) if QB_CORREGIR_JUEGA else 0.0
    return dict(n=q["n"], bk=bk, est=est, delta=round(float(dl), 1), s=round(float(s), 2),
                estado=estado, juega=round(float(juega), 2), fuera=round(float(-s * dl), 2))


def ajuste_base(s):
    """Modo QB_OUT_ONLY: solo ajusta cuando el reporte confirma que el QB está fuera/inactivo; el reemplazo se selecciona entre los QB recientes disponibles.
    Si está activo, dudoso o no aparece en el reporte, deja intacta la proyección Ridge."""
    return 0.0 if not s or s["estado"] != "fuera" else float(s["fuera"])


def aplicar_qb(mar, tot, adj_h, adj_a):
    ph, pa = (tot + mar) / 2 + adj_h, (tot - mar) / 2 + adj_a
    return ph - pa, ph + pa


def alternativas_qb(h, a, sh, sa, mar0, tot0, spread_line, total_line):
    """Un escenario 'si no juega' por cada QB DUDOSO. Va al final de la tarjeta."""
    alt = []
    for eq, s, lado in ((h, sh, "h"), (a, sa, "a")):
        if not s or s["estado"] != "duda":
            continue
        ah = s["fuera"] if lado == "h" else ajuste_base(sh)
        aa = s["fuera"] if lado == "a" else ajuste_base(sa)
        m2, t2 = aplicar_qb(mar0, tot0, ah, aa)
        ape = s["n"].split()[-1]
        alt.append(dict(qb=s["n"], eq=eq, est=s["est"], pen=s["fuera"], bk=s["bk"],
                        proy=[round((t2 + m2) / 2, 1), round((t2 - m2) / 2, 1)],
                        p_h=float(norm_cdf(m2 / SD_MARGEN)),
                        rows=[fila(f"Sin {ape} · Spread · local gana por >", m2, "n", SD_MARGEN, num(spread_line),
                                   f"{h} cubre", f"{a} cubre"),
                              fila(f"Sin {ape} · Total de puntos", t2, "n", 13, num(total_line))]))
    return alt


def filas_nfl(h, a, mar, tot, sp_line, tot_line, M, ref, R=None, bad=None, duda=None, adj=(0.0, 0.0)):
    """bad: {equipo: nombres que NO jugarán (se excluyen)}. duda: {equipo: {nombre: estatus}} dudables: la fila de quien
    SÍ juega va arriba (etiqueta DUDA) y la de su suplente (si no juega) va hasta abajo.
    El jugador se toma del depth chart de ESPN (saltando lesionados); en Resultados (R) no se usa el depth chart actual.
    adj: ajuste de puntos por QB (local, visita) que también mueve las yardas aéreas del equipo."""
    R = R or {}
    bad = bad or {}
    duda = duda or {}
    usar_depth = not R
    rows = [
        fila("Spread · local gana por >", mar, "n", SD_MARGEN, num(sp_line), f"{h} cubre", f"{a} cubre", real=R.get("spread")),
        fila("Total de puntos", tot, "n", 13, num(tot_line), real=R.get("total")),
    ]
    dudosos = []
    if M.get("air"):
        eah, eaa = esperado(M["air"], h, a)
        eah *= 1 + QB_YDS_K * adj[0] / PTS_EQUIPO
        eaa *= 1 + QB_YDS_K * adj[1] / PTS_EQUIPO
        egh, ega = esperado(M["gr"], h, a)
        rows += [
            fila(f"Yds aire {h}", eah, "n", M["sd_air"], real=R.get("air_h")),
            fila(f"Yds aire {a}", eaa, "n", M["sd_air"], real=R.get("air_a")),
            fila(f"Yds suelo {h}", egh, "n", M["sd_gr"], real=R.get("gr_h")),
            fila(f"Yds suelo {a}", ega, "n", M["sd_gr"], real=R.get("gr_a")),
        ]
        for team, opp in ((h, a), (a, h)):
            for pos, etq, key in (("QB", "yds pase", "air"), ("RB", "yds suelo", "gr"), ("WR", "yds recep.", "air")):
                dfn = M[key][1].get(opp, 1)
                fuera = bad.get(team, frozenset())
                j = jugador(M, team, pos, ref, dfn, fuera, depth=usar_depth)
                if not j:
                    continue
                pid, nombre, mu, sd = j
                real = None
                if "ps_game" in R:
                    sub = R["ps_game"]
                    v = sub.loc[sub.player_id == pid, POS[pos][0]]
                    real = float(v.iloc[0]) if len(v) else None
                est = duda.get(team, {}).get(_nj(nombre))
                if not est:
                    rows.append(fila(f"{pos} {nombre} ({team}) · {etq}", mu, "n", sd, real=real))
                    continue
                # dudable: SI JUEGA va arriba con su valor normal (etiqueta DUDA)...
                rows.append(fila(f"{pos} {nombre} ({team}) · {etq} · ⚠ DUDA", mu, "n", sd, real=real))
                # ...y SI NO JUEGA va hasta abajo con el siguiente del depth chart
                excl = fuera | frozenset(duda.get(team, {})) | {_nj(nombre)}
                s2 = jugador(M, team, pos, ref, dfn, excl, activo=False, depth=usar_depth)
                if s2:
                    _, n2, mu2, sd2 = s2
                    dudosos.append(fila(f"Si no juega {nombre.split()[-1]} → {pos} {n2} ({team}) · {etq}", mu2, "n", sd2, real=None))
        if dudosos:
            dudosos[0]["sep"] = True      # el HTML dibuja una línea y el rótulo "Si no juega" antes de esta fila
            rows += dudosos
    if R:
        rows = [r for r in rows if r["real"] is not None]
    return rows


def hora_nfl_gdl(fecha, hora):
    """Convierte kickoff de NFL (hora del Este) a hora del centro de México."""
    try:
        if not hora or pd.isna(hora):
            return ""
        base = pd.Timestamp(f"{pd.Timestamp(fecha).date()} {str(hora).strip()}")
        return base.tz_localize("America/New_York").tz_convert("America/Mexico_City").strftime("%H:%M")
    except Exception:
        return ""


def nfl():
    import nflreadpy as nfr
    temp = HOY.year if HOY.month >= 3 else HOY.year - 1
    seasons = [temp - 1, temp]
    sch = nfr.load_schedules(seasons).to_pandas()
    sch["gameday"] = pd.to_datetime(sch["gameday"])
    sch["home_team"] = sch["home_team"].replace(NFL_TEAM)
    sch["away_team"] = sch["away_team"].replace(NFL_TEAM)
    jug = sch.dropna(subset=["home_score", "away_score"])
    gl_nfl = jug[["gameday", "home_team", "away_team", "home_score", "away_score"]] \
        .rename(columns={"gameday": "Date", "home_team": "h", "away_team": "a", "home_score": "gh", "away_score": "ga"})

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

    # ── Lesiones por semana de cada juego + modelo de QB ──
    inj_w = {}
    if len(prox):
        try:
            inj_w = lesionados(nfr, temp, {int(w) for w in prox.week})
        except Exception as e:
            diag(f"NFL lesiones falló: {e}")
    Q = qb_modelo(ps, ref_hoy)
    SALARIOS_QB = nfl_qb_salarios(nfr) if len(prox) else {}
    if len(prox):
        nfl_qb_roster(nfr, temp)
    if len(prox):
        nfl_qb_espn(temp)                                      # ESPN pisa roster y salarios de nflverse
        nfl_depth()                                            # depth chart de ESPN (QB / RB / WR)
    if Q:
        top = sorted(Q.items(), key=lambda kv: -kv[1]["delta"])[:5]
        diag("QB · valor vs suplente (pts/juego), top 5: " + ", ".join(f"{t} {q['n']} {q['delta']:.1f}" for t, q in top))
    forma = estrellas_liga(gl_nfl, ref_hoy)

    out = []
    for r in prox.itertuples():
        h, a = r.home_team, r.away_team
        inj = inj_w.get(int(r.week), {})
        bad, duda = partir_lesiones(inj)
        for t_ in (h, a):
            diag(f"Depth {t_}: QB {DEPTH.get(t_, {}).get('QB', [])[:3]} · RB {DEPTH.get(t_, {}).get('RB', [])[:2]} · WR {DEPTH.get(t_, {}).get('WR', [])[:3]} · fuera {sorted(bad.get(t_, []))}")

        mar0, tot0 = pred_nfl(M_hoy, h, a)                     # modelo base (sin ajuste de QB)
        sh = escenario_qb(Q, h, inj, M_hoy, ref_hoy, SALARIOS_QB)
        sa = escenario_qb(Q, a, inj, M_hoy, ref_hoy, SALARIOS_QB)
        ah, aa = ajuste_base(sh), ajuste_base(sa)
        mar, tot = aplicar_qb(mar0, tot0, ah, aa)              # modelo normal (QB dudoso = juega)
        alt = alternativas_qb(h, a, sh, sa, mar0, tot0, r.spread_line, r.total_line)
        qbn = [dict(eq=eq_, n=s["n"], estado=s["estado"], est=s["est"], adj=ajuste_base(s), fuera=s["fuera"], bk=s["bk"])
               for eq_, s in ((h, sh), (a, sa)) if s and s["estado"] != "ok"]

        rows = filas_nfl(h, a, mar, tot, r.spread_line, r.total_line, M_hoy, ref_hoy, bad=bad, duda=duda, adj=(ah, aa))
        ph = norm_cdf(mar / SD_MARGEN)
        probs = [dict(l=h, p=float(ph), c=ml_dec(getattr(r, "home_moneyline", None))),
                 dict(l=a, p=float(1 - ph), c=ml_dec(getattr(r, "away_moneyline", None)))]
        les = dict(h=bajas_clave(M_hoy, h, ref_hoy, inj), a=bajas_clave(M_hoy, a, ref_hoy, inj)) if inj else None
        out.append(dict(sp="NFL", lg="NFL", date=str(r.gameday.date()), hora=hora_nfl_gdl(r.gameday, getattr(r, "gametime", "")), home=h, away=a,
                        week=int(r.week), div=es_div(r, h, a),
                        logo_h=logo(h, "NFL"), logo_a=logo(a, "NFL"),
                        proy=[round((tot + mar) / 2, 1), round((tot - mar) / 2, 1)],
                        racha=dict(h=racha_equipo(gl_nfl, h, ref_hoy), a=racha_equipo(gl_nfl, a, ref_hoy)),
                        les=les, forma=dict(h=forma.get(h), a=forma.get(a)),
                        qb=qbn, alt=alt,
                        rows=rows, probs=probs, ia=""))

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


def odds_comp(c):
    """Momios 1X2 (decimales) que ESPN adjunta a un partido, si la casa ya los publicó: [local, empate, visita].
    Nunca lanza error: si el formato es raro devuelve [None]*3 (un fallo aquí no debe tirar el partido)."""
    try:
        odds = c.get("odds") or []
        if isinstance(odds, dict):
            odds = [odds]
        for o in odds:
            if not isinstance(o, dict):
                continue
            ml = o.get("moneyline") if isinstance(o.get("moneyline"), dict) else {}
            v = []
            for k in ("home", "draw", "away"):
                x = ml.get(k)
                x = x.get("close") if isinstance(x, dict) else None
                v.append(ml_dec(x.get("odds") if isinstance(x, dict) else None))
            if all(v):
                return v
            v = []
            for k in ("homeTeamOdds", "drawOdds", "awayTeamOdds"):
                x = o.get(k)
                v.append(ml_dec(x.get("moneyLine") if isinstance(x, dict) else None))
            if all(v):
                return v
    except Exception:
        pass
    return [None] * 3


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
            out.append(dict(Date=pd.to_datetime(e["date"], utc=True).tz_convert("America/Mexico_City").tz_localize(None),
                            Hora=pd.to_datetime(e["date"], utc=True).tz_convert("America/Mexico_City").strftime("%H:%M"),
                            h=H["team"]["displayName"], a=Av["team"]["displayName"],
                            FTHG=num(H.get("score")), FTAG=num(Av.get("score")),
                            HC=stat_espn(H, ("wonCorners",)), AC=stat_espn(Av, ("wonCorners",)),
                            HST=stat_espn(H, ("shotsOnTarget",)), AST=stat_espn(Av, ("shotsOnTarget",)),
                            cu=odds_comp(c),
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


def odds_espn(slug, nombres, dias):
    """Respaldo de momios: los que ESPN publica en el scoreboard de los días indicados.
    Devuelve {(local, visita): [1, X, 2]} con los nombres ya mapeados al histórico."""
    out = {}
    dias = sorted({d + dt.timedelta(days=k) for d in dias for k in (-1, 0)})   # ±1 día por la zona horaria

    def uno(d):
        try:
            return _espn_dia(slug, d)
        except Exception:
            return []

    with ThreadPoolExecutor(8) as ex:
        eventos = [e for lst in ex.map(uno, dias) for e in lst]
    for e in eventos:
        try:
            c = e["competitions"][0]
            H = next(x for x in c["competitors"] if x["homeAway"] == "home")
            Av = next(x for x in c["competitors"] if x["homeAway"] == "away")
            cu = odds_comp(c)
            if all(cu):
                out[(mapear(H["team"]["displayName"], nombres), mapear(Av["team"]["displayName"], nombres))] = cu
        except Exception:
            continue
    diag(f"{slug}: ESPN trae momios de {len(out)} partidos")
    return out


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
                    res.append(dict(h=h2, a=a2, Date=r.Date, Hora=getattr(r, "Hora", ""), cu=list(r.cu), src="ESPN"))
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
            f_utc = pd.to_datetime(m["utcDate"], utc=True)
            f_local = f_utc.tz_convert("America/Mexico_City")
            f = f_local.tz_localize(None)
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
            res.append(dict(h=h2, a=a2, Date=f, Hora=f_local.strftime("%H:%M"), cu=[None] * 3, src="football-data.org"))
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
        diag(f"{sport}: The Odds API omitida (falta el secreto ODDS_API_KEY)")
        return {}
    try:
        r = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport}/odds/",
                         params=dict(apiKey=k, regions="eu,uk,us", markets="h2h", oddsFormat="decimal"), timeout=30)
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
            # Momios: fixtures.csv → The Odds API → ESPN. Solo se consulta ESPN para los partidos que aún no tienen
            faltan_d = {p["Date"].date() for p in prox if not all(p["cu"]) and (p["h"], p["a"]) not in odds}
            odds_esp = odds_espn(slug, nombres, faltan_d) if faltan_d else {}
            cargar_logos(slug, nombres)

            # Racha y estrellas (últimos N_FORMA juegos de cada equipo)
            gl = hist[["Date", "h", "a", "FTHG", "FTAG"]].rename(columns={"FTHG": "gh", "FTAG": "ga"}).copy()
            gl["gh"], gl["ga"] = pd.to_numeric(gl.gh, errors="coerce"), pd.to_numeric(gl.ga, errors="coerce")
            gl = gl.dropna(subset=["gh", "ga"])
            forma = estrellas_liga(gl, ref_hoy)

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
                    if not all(cu):
                        cu = odds.get((h, a)) or odds_esp.get((h, a)) or cu
                    con_cuota += 1 if all(cu) else 0
                    probs = [dict(l=l, p=float(pp), c=c) for l, pp, c in zip((h, "Empate", a), (p1, px, p2), cu)]
                    out.append(dict(sp="FUT", lg=liga, date=str(p["Date"].date()), hora=p.get("Hora", ""), home=h, away=a,
                                    logo_h=logo(h, "FUT"), logo_a=logo(a, "FUT"),
                                    proy=[round(lh, 1), round(la, 1)], rows=rows, probs=probs, ia="",
                                    racha=dict(h=racha_equipo(gl, h, ref_hoy), a=racha_equipo(gl, a, ref_hoy)),
                                    forma=dict(h=forma.get(h), a=forma.get(a))))
                except Exception as e:
                    diag(f"{liga} próximo {p.get('h')}-{p.get('a')}: {e}")
            diag(f"{liga}: {con_cuota} de {len(prox)} partidos con momios 1X2"
                 + ("" if con_cuota == len(prox) else " (las casas publican ~1-2 semanas antes; el resto usa el momio justo del modelo)"))
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
<link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' fill='%23090912'/%3E%3Ctext x='50' y='64' font-size='46' text-anchor='middle' fill='%23d6264f' font-family='Arial' font-weight='900'%3E%2F%2F%2F%3C/text%3E%3C/svg%3E">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#090912;--card:#11112a;--line:#2a1b3a;--red:#d6264f;--tx:#e9e9f2;--mut:#8a8aa3;--ok:#3c7a4a;--wrong:#e03e3e}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html{-webkit-text-size-adjust:100%;text-size-adjust:100%}
body{margin:0;overflow-x:hidden;min-height:100vh;background:radial-gradient(1200px 500px at 50% -100px,#2a0f26 0,transparent 70%),var(--bg);background-attachment:fixed;color:var(--tx);font-family:Barlow,-apple-system,sans-serif;padding-bottom:env(safe-area-inset-bottom)}
header.top{display:block;position:sticky;top:0;z-index:50;background:rgba(9,9,18,.88);-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);padding-top:env(safe-area-inset-top)}
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
nav{display:flex;flex-wrap:wrap;align-items:center;gap:6px 30px;padding:12px 40px;border-bottom:1px solid #1d1530;font-family:'Barlow Condensed';letter-spacing:.14em;font-weight:700;overflow:visible}
nav::-webkit-scrollbar{display:none}
.logo{order:1;font-size:26px;line-height:1.25;letter-spacing:.06em;cursor:pointer;flex:none;white-space:nowrap}.logo i{color:var(--red);font-style:normal}
nav a{cursor:pointer;color:var(--mut);font-size:15px;padding:8px 0;transition:.2s;flex:none}nav a.on{color:var(--red);border-bottom:2px solid var(--red)}
.fmt{order:3;margin-left:auto;flex:none;background:transparent;border:1px solid var(--red);color:var(--red);border-radius:20px;padding:8px 16px;cursor:pointer;font:700 13px 'Barlow Condensed';letter-spacing:.14em;text-transform:uppercase;-webkit-appearance:none}
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
/* el icono va en el flujo (no absoluto): así nunca se encima con el título; margin-bottom:auto lo mantiene arriba */
.tic{position:relative;margin-bottom:auto;padding-bottom:14px;font-size:46px;line-height:1;filter:drop-shadow(0 4px 12px rgba(0,0,0,.5))}
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
.vs{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}.vs .sc,.vs .scw{align-self:center}.eq{text-align:center;flex:1;min-width:0}
.ci{width:64px;height:64px;border-radius:50%;background:linear-gradient(135deg,#1b0f2e,#11112a);border:1px solid var(--line);margin:0 auto 8px;display:grid;place-items:center;overflow:hidden;font:800 20px 'Barlow Condensed';color:var(--mut)}
.ci img{width:80%;height:80%;object-fit:contain}
.eq b{font:700 18px 'Barlow Condensed';text-transform:uppercase;display:block;margin-top:4px}.sc{font:800 44px 'Barlow Condensed';white-space:nowrap}
.scw{text-align:center}.pj{font:700 12px 'Barlow Condensed';letter-spacing:.14em;color:var(--mut);text-transform:uppercase;white-space:nowrap}
.pj.p{color:#c9c9dc;margin-bottom:2px}
/* racha y bajas (debajo de cada equipo) */
.rt{font:700 10px 'Barlow Condensed';letter-spacing:.16em;color:var(--mut);margin-top:8px}
.rc{display:flex;gap:3px;justify-content:center;margin-top:3px}
.rc i{font:800 10px 'Barlow Condensed';font-style:normal;width:17px;height:17px;border-radius:4px;display:grid;place-items:center;color:#fff}
.rc i.G{background:var(--ok)}.rc i.P{background:var(--wrong)}.rc i.E{background:#555570}
.les{margin-top:6px;font-size:11px;line-height:1.35;color:#ff9a9a;overflow-wrap:break-word}
.les.ok{color:#6fd08a}
.pr{display:flex;height:6px;border-radius:3px;overflow:hidden;margin:16px 0 4px}
.pl{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);gap:6px}
/* desplegable de momios */
.mbtn{width:100%;max-width:100%;margin-top:16px;display:flex;justify-content:space-between;align-items:center;background:#0e0e24;border:1px solid var(--line);border-radius:10px;color:var(--tx);padding:12px 14px;cursor:pointer;font:800 13px 'Barlow Condensed';letter-spacing:.16em;text-transform:uppercase;-webkit-appearance:none;appearance:none}
.mbtn:hover{border-color:var(--red)}
.ar{display:inline-block;color:var(--red);font-size:16px;transition:transform .25s}
.mx.open .ar{transform:rotate(180deg)}.mx.open .mbtn{border-color:var(--red)}
.mbody{display:none}.mx.open .mbody{display:block}
.edge,.fm{overflow:hidden;margin-top:14px;border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:#0e0e24}
.edge h4,.fm h4{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 8px;font:800 13px 'Barlow Condensed';letter-spacing:.16em;color:var(--red)}
.fmt2{flex:none;background:transparent;border:1px solid var(--red);color:var(--red);border-radius:16px;padding:5px 12px;cursor:pointer;font:700 12px 'Barlow Condensed';letter-spacing:.12em;text-transform:uppercase;-webkit-appearance:none;appearance:none}
.fmt2:active{background:var(--red);color:#fff}
.eo{display:grid;grid-template-columns:minmax(0,1fr) auto auto 64px;gap:8px;align-items:center;font-size:13px;padding:3px 0}.eo>*{min-width:0}.eo>span:first-child{overflow-wrap:anywhere}
.eo .fj{color:var(--mut);font-size:12px;white-space:nowrap}.eo .ed{font-weight:700;text-align:right}
.nm{color:var(--mut);font-size:11px;line-height:1.4;margin-top:6px}
/* estrellas ofensiva / defensiva */
.fb b{font:700 14px 'Barlow Condensed';letter-spacing:.06em;text-transform:uppercase;display:block;margin:8px 0 2px}
.fr{display:grid;grid-template-columns:70px auto minmax(0,1fr);gap:8px;align-items:center;font-size:12px;padding:2px 0}
.fr span:last-child{color:var(--mut);font-size:11px}
.st{color:#f5b301;letter-spacing:1px;font-size:14px;white-space:nowrap}.st s{text-decoration:none;opacity:.2}
table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px}td{padding:6px 4px;border-top:1px solid #1d1d3a}td:nth-child(2){color:var(--mut)}
tr.sp td{border-top:2px solid var(--red);padding-top:10px;color:var(--red);font:800 11px 'Barlow Condensed';letter-spacing:.14em;text-transform:uppercase;white-space:normal}
tr.hd td{color:var(--mut);font:700 11px 'Barlow Condensed';letter-spacing:.12em;text-transform:uppercase;border-top:0;white-space:nowrap}
input{width:62px;background:#0c0c1e;border:1px solid var(--line);border-radius:6px;color:var(--tx);padding:5px 6px;-webkit-appearance:none;appearance:none}
.eo input{width:70px}
/* campo + botón ± (el teclado decimal de iOS no trae el signo menos) */
.ib{display:inline-flex;gap:3px;align-items:center;white-space:nowrap}
.sg{flex:none;width:26px;height:30px;padding:0;background:transparent;border:1px solid var(--red);color:var(--red);border-radius:6px;font:800 15px 'Barlow Condensed';cursor:pointer;-webkit-appearance:none;appearance:none}
.sg:active{background:var(--red);color:#fff}
.g{color:#6fd08a;font-weight:500}
.ctx{margin-top:12px;color:var(--mut);font-size:12px;border-left:2px solid var(--red);padding-left:10px}
.ai{border:1px solid #6b2a5a;background:linear-gradient(135deg,#1b0f2e,#2a0f26);border-radius:14px;padding:18px 22px;margin-bottom:22px;line-height:1.55;color:#e9d5ff}.ai h3{margin:0 0 6px;font:800 16px 'Barlow Condensed';letter-spacing:.16em;color:var(--red)}
.pks{border:1px solid #6b2a5a;background:linear-gradient(135deg,#1b0f2e,#2a0f26);border-radius:14px;padding:18px 22px;margin-bottom:22px}
.pks h3{margin:0 0 8px;font:800 16px 'Barlow Condensed';letter-spacing:.16em;color:var(--red)}
.pk{display:flex;gap:12px;align-items:flex-start;padding:10px 0;border-top:1px solid #3a1f4a;font-size:14px;line-height:1.45}
.pk .cf{font:800 12px 'Barlow Condensed';letter-spacing:.12em;border-radius:12px;padding:2px 10px;white-space:nowrap;background:#555570}
.pk .cf.alta{background:var(--ok)}.pk .cf.media{background:#b8861b}
.pk small{color:var(--mut);display:block}.pk b{color:#fff}
.tabs{order:2;width:auto;display:flex;gap:30px;flex:none;overflow-x:auto;-webkit-overflow-scrolling:touch;scrollbar-width:none}.tabs::-webkit-scrollbar{display:none}.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}.tw table{margin-top:12px}
td{overflow-wrap:break-word}td:nth-child(2),td:nth-child(3){white-space:nowrap}td:nth-child(4){min-width:112px}
.tag{background:#2a1b3a;color:#e9d5ff;border-radius:8px;padding:1px 8px;margin-left:6px;font-size:11px}
.diag{color:var(--mut);font-size:12px;line-height:1.6;margin-top:10px;white-space:pre-wrap}
/* aviso de QB y escenario "si no juega" */
.qbn{margin-top:12px;font-size:12px;line-height:1.5;color:#ffd27a;border-left:2px solid #f5b301;padding-left:10px}
.alt{margin-top:16px;border:1px dashed #6b2a5a;border-radius:10px;padding:10px 12px;background:#0e0e24}
.alt h4{margin:0 0 6px;font:800 13px 'Barlow Condensed';letter-spacing:.14em;color:#f5b301}
.nota{color:var(--mut);font-size:12px;margin-top:26px}
.updated{padding:7px 22px 9px;text-align:right;color:var(--mut);font:600 10px 'Barlow Condensed';letter-spacing:.12em}
@media(pointer:coarse){input,.eo input{font-size:16px}}   /* evita el zoom automático de Safari al escribir */
@media(max-width:640px){
 main{padding:20px 16px calc(24px + env(safe-area-inset-bottom))}nav{padding:10px 16px;gap:20px}h1{font-size:40px}
 .card{padding:18px 16px}.sc{font-size:34px}.eq b{font-size:15px}.ci{width:52px;height:52px}
 nav{flex-wrap:wrap;overflow:visible;row-gap:6px}.logo{order:1;font-size:22px}.fmt{order:2;margin-left:auto;padding:6px 12px}.tabs{order:3;width:100%;}
 .tabs{order:3;width:100%;gap:22px;overflow-x:auto;-webkit-overflow-scrolling:touch;scrollbar-width:none}.tabs::-webkit-scrollbar{display:none}
 .eo{grid-template-columns:minmax(0,1fr) auto 58px;gap:6px}.eo .fj{grid-column:1/-1;order:3;white-space:normal}.eo input{width:70px}
 table{font-size:12px}td{padding:6px 3px}.g{font-size:11px}.tw input{width:54px}
 .tile{min-height:190px;padding:20px}.tile h2{font-size:34px}.tic{font-size:38px}
 .tbg img:nth-child(n+3){display:none}
 .rc i{width:15px;height:15px;font-size:9px}.les{font-size:10px}
 .fr{grid-template-columns:62px auto minmax(0,1fr)}
}
</style></head><body>
<header class="top">
<div class="tick" id="tick"></div>
<nav><div class="logo" onclick="setTab('HOME')">EDGE<i>///</i>LAB</div>
<div class="tabs"><a id="tNFL" onclick="setTab('NFL')">NFL</a>
<a id="tFUT" onclick="setTab('FUT')">FÚTBOL</a>
<a id="tCL" onclick="setTab('CL')">CHAMPIONS</a>
<a id="tRES" onclick="setTab('RES')">RESULTADOS</a></div>
<button class="fmt" id="fmtbtn" onclick="toggleFmt()"></button>
</nav>
<div class="updated">ÚLTIMA ACTUALIZACIÓN: __ACTUALIZADO__ · HORA DE GUADALAJARA</div>
</header>
<main><h1 id="ttl"></h1><div id="descr" class="sub"></div>
<div id="home" class="lobby"></div>
<div id="ai"></div><div id="picks"></div><div class="chips" id="chips"></div><div class="grid" id="grid"></div><div class="nota">Modelo estadístico (Ridge / Poisson / Normal). Edge = probabilidad del modelo × cuota decimal − 1 (el momio de la casa ya incluye su margen). Puedes escribir tu momio en americano (+150, -110) o decimal (2.50): se detecta solo; con el botón ± cambias el signo. 🏠 = equipo local. Racha = últimos 5 juegos (G ganó · E empató · P perdió). Estrellas 1-5 = ranking de todos los equipos de la liga con sus últimos 5 juegos: puntos/goles a favor (ofensiva) y en contra (defensiva), ajustados por la calidad del rival; 5 = el 20 % mejor. Bajas = reporte de lesiones de ESPN/nflverse (gana el reporte más reciente); los jugadores que no jugarán (Out, IR, PUP, suspendidos) se excluyen de las apuestas y se reemplazan por el siguiente del depth chart de ESPN (QB, RB y WR); si ESPN no trae el depth chart se usa el volumen reciente. Los dudables (Questionable, Doubtful, Day-To-Day) aparecen marcados con DUDA: su fila normal va arriba y la del suplente del depth chart (si no juega) hasta abajo. Ajuste de QB: el valor del mariscal titular sobre su suplente se mide con EPA por jugada; si está fuera o dudoso, el reemplazo es el siguiente QB del depth chart de ESPN que no aparezca en el reporte de lesiones; si ESPN no lo trae, se elige por mayor salario base NFL de la temporada actual (contratos OverTheCap vía nflverse) y, si no hay salario utilizable, por uso reciente. El salario se expresa en millones de dólares por año. El modelo ya resta esos puntos, y si está dudoso la tabla asume que juega y abajo aparece el escenario "si no juega". Cifras redondeadas, salvo proyecciones, spread y total. Los picks de Gemini combinan tu modelo con noticias de la web y pueden fallar. Solo informativo, no es asesoría financiera. En Resultados se visualizan exclusivamente filas con datos reales.</div></main>"""

HTML_PART2_JS = r"""<script>
const D = __DATA__;
const HASH = {home:'HOME', nfl:'NFL', fut:'FUT', cl:'CL', res:'RES'};
let tab = HASH[(location.hash || '').slice(1)] || 'HOME', lg = 'Todas', FMT = 'us';
const ROWS = [], ROWK = [], PR = [], PRK = [], CK = [];
const ODDS = {}, LINES = {}, OPEN = new Set();
const TT = {NFL:'NFL', FUT:'Fútbol', CL:'Champions', RES:'Resultados'};
const SUB = {NFL:'Semana actual · modelo vs. mercado', FUT:'La Liga y Premier League', CL:'UEFA Champions League', RES:'Lo que proyectó el modelo vs. lo que pasó realmente'};
const EST = {'out':'FUERA', 'doubtful':'DUDOSO', 'questionable':'DUDA', 'injured reserve':'IR', 'ir':'IR', 'day-to-day':'DÍA A DÍA', 'pup':'PUP', 'suspension':'SUSP.'};

const toUS = d => d >= 2 ? '+' + Math.round((d - 1) * 100) : '-' + Math.round(100 / (d - 1));
const fmtO = d => (d == null || !isFinite(d) || d <= 1) ? '' : (FMT === 'us' ? toUS(d) : d.toFixed(2));
const f1 = x => (x == null || isNaN(x)) ? '' : Number(x).toFixed(1);   // proyecciones, spread y total: 1 decimal
const f0 = x => (x == null || isNaN(x)) ? '' : Math.round(x);          // todo lo demás: entero
const KEEP = l => /Spread|Total de puntos|Goles totales/.test(l);      // filas que conservan 1 decimal
function parseO(s) {
  s = String(s).trim().replace(',', '.');
  if (s === '') return null;
  const v = parseFloat(s);
  if (isNaN(v)) return null;
  if (Math.abs(v) >= 100) return v > 0 ? 1 + v / 100 : 1 + 100 / Math.abs(v);
  return v > 1 ? v : null;
}
function toggleFmt() { FMT = FMT === 'us' ? 'dec' : 'us'; render(); }

// botón ±: cambia el signo del campo anterior (el teclado decimal del teléfono no tiene "-")
function neg(b) {
  const i = b.parentNode.querySelector('input');
  let v = i.value.trim();
  v = v === '' ? '-' : (v[0] === '-' ? v.slice(1) : '-' + v);
  i.value = v;
  i.dispatchEvent(new Event('input'));
  i.focus();
}

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
  document.getElementById('o' + i).textContent = v === '' ? '' : txt(ROWS[i], parseFloat(v.replace(',', '.')));
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
    el.innerHTML = `<span style="color:${e >= 0 ? '#6fd08a' : '#e03e3e'}">${e >= 0 ? '+' : ''}${(e * 100).toFixed(0)}%</span>`;
  });
  const b = document.getElementById('b' + gid);
  if (b) b.textContent = (best != null && best > 0) ? `VALOR +${(best * 100).toFixed(0)}%` : '';
}
function setOdd(gid, k, val) { ODDS[PRK[gid] + '|' + k] = parseO(val); edge(gid); }

function rowH(r, hist, mk) {
  const i = ROWS.push(r) - 1;
  ROWK[i] = mk + '|' + r.l;
  const esP = r.p != null || r.o === 'Sí';                       // filas de probabilidad (Ambos anotan)
  const mv = esP ? Math.round(r.m * 100) + '%' : (KEEP(r.l) ? f1(r.m) : f0(r.m));   // modelo: entero (1 decimal solo spread/total)
  if (hist) return `<tr><td>${r.l}</td><td>${mv}</td><td>${r.o === 'Sí' ? (r.real ? 'Sí' : 'No') : (r.real ?? '–')}</td></tr>`;
  const lv = ROWK[i] in LINES ? LINES[ROWK[i]] : (r.line ?? '');
  const inp = r.p != null ? '' : `<span class="ib"><input id="i${i}" type="text" inputmode="decimal" autocomplete="off" autocapitalize="off" spellcheck="false" value="${lv}" oninput="upd(${i})"><button type="button" class="sg" onclick="neg(this)">±</button></span>`;
  const sp = r.sep ? '<tr class="sp"><td colspan="4">Si no juega el dudoso · alternativa</td></tr>' : '';
  return `${sp}<tr><td>${r.l}</td><td>${mv}</td><td>${inp}</td><td class="g" id="o${i}">${txt(r, lv === '' ? null : parseFloat(String(lv).replace(',', '.')))}</td></tr>`;
}
const ini = n => n.split(/\s+/).map(w => w[0]).join('').slice(0, 3).toUpperCase();
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// equipo + racha (últimos 5) + bajas clave debajo
function team(n, logo, loc, rc, les) {
  const img = logo ? `<img src="${logo}" onerror="this.parentNode.textContent='${ini(n).replace(/'/g, '')}'">` : ini(n);
  const chips = rc ? `<div class="rt">RACHA</div><div class="rc">${[...rc].map(c => `<i class="${c}">${c}</i>`).join('')}</div>` : '';
  let lh = '';
  if (les) lh = les.length
    ? `<div class="les">${les.map(x => `⚠ ${esc(x.pos)} ${esc(x.n)} · ${EST[String(x.est).toLowerCase()] || esc(x.est)}`).join('<br>')}</div>`
    : '<div class="les ok">✓ Sin bajas clave</div>';
  return `<div class="eq"><div class="ci">${img}</div><b>${loc ? '🏠 ' : ''}${n}</b>${chips}${lh}</div>`;
}

const star = n => `<span class="st">${'★'.repeat(n)}<s>${'★'.repeat(5 - n)}</s></span>`;
function formaH(g) {
  if (!g.forma) return '';
  const u = g.sp === 'NFL' ? 'pts' : 'goles';
  const b = (nom, f) => f ? `<div class="fb"><b>${esc(nom)}</b>
    <div class="fr"><span>Ofensiva</span>${star(f.of)}<span>${f0(f.gf)} ${u}/juego</span></div>
    <div class="fr"><span>Defensiva</span>${star(f.de)}<span>${f0(f.gc)} ${u} en contra</span></div></div>` : '';
  const x = b(g.home, g.forma.h) + b(g.away, g.forma.a);
  return x ? `<div class="fm"><h4><span>FORMA · ÚLTIMOS 5 JUEGOS</span></h4>${x}</div>` : '';
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
    : `<div class="sc">${f1(g.proy[0])} - ${f1(g.proy[1])}</div>`;
  const badge = hist ? `<span class="badge ${g.atino === 'SI' ? '' : 'incorrecto'}">${g.atino === 'SI' ? 'ACERTÓ' : 'FALLÓ'}</span>` : '';

  let pr = '', edgeH = '', valB = '';
  if (g.probs && !hist) {
    const gid = PR.push(g.probs) - 1;
    PRK[gid] = mk;
    const cols = g.probs.length === 2 ? ['#d6264f', '#6b8cff'] : ['#d6264f', '#555570', '#6b8cff'];
    pr = `<div class="pr">${g.probs.map((x, k) => `<div style="width:${(x.p * 100).toFixed(1)}%;background:${cols[k]}"></div>`).join('')}</div>
          <div class="pl">${g.probs.map(x => `<span>${x.l} ${(x.p * 100).toFixed(0)}%</span>`).join('')}</div>`;
    valB = `<span class="val" id="b${gid}"></span>`;
    const sinC = g.probs.every(x => x.c == null);
    edgeH = `<div class="edge"><h4><span>EDGE · RESULTADO</span><button class="fmt2" onclick="toggleFmt()">${FMT === 'us' ? 'Americano' : 'Decimal'} ⇄</button></h4>${g.probs.map((x, k) => {
      const key = mk + '|' + k;
      const d = key in ODDS ? ODDS[key] : x.c;
      const pj = Math.min(0.995, Math.max(0.005, x.p));
      return `<div class="eo"><span>${x.l}</span><span class="fj">${(x.p * 100).toFixed(0)}% · justo ${fmtO(1 / pj)}</span>
        <span class="ib"><input type="text" inputmode="decimal" autocomplete="off" autocapitalize="off" spellcheck="false" placeholder="momio" value="${fmtO(d)}" oninput="setOdd(${gid},${k},this.value)"><button type="button" class="sg" onclick="neg(this)">±</button></span>
        <span class="ed" id="e${gid}_${k}"></span></div>`;
    }).join('')}${sinC ? '<div class="nm">Las casas aún no publican momios de este partido (suelen salir 1-2 semanas antes). Usa el momio justo del modelo como referencia o escribe el tuyo para ver el edge.</div>' : ''}</div>`;
  }
  const head = hist ? '<tr class="hd"><td>Métrica</td><td>Proyectado</td><td>Real</td></tr>' : '<tr class="hd"><td></td><td>Modelo</td><td>Tu línea</td><td>Prob · justo</td></tr>';
  const ctx = hist ? '' : (g.ia ? `<div class="ctx">${g.ia}</div>` : '');
  const etq = hist ? 'Proyección vs. real' : 'Momios y mercados';

  // aviso del QB (visible siempre) y escenario "si no juega" (hasta abajo del desplegable)
  const EQ = q => q.estado === 'fuera' ? 'FUERA' : (EST[String(q.est).toLowerCase()] || esc(q.est));
  const qbH = (!hist && g.qb && g.qb.length) ? `<div class="qbn">${g.qb.map(q => q.estado === 'fuera'
    ? `🚑 QB ${esc(q.n)} (${esc(q.eq)}) · ${EQ(q)} → el modelo ya le resta ${Math.abs(q.adj).toFixed(1)} pts a ${esc(q.eq)} (entra ${esc(q.bk)})`
    : `⚠ QB ${esc(q.n)} (${esc(q.eq)}) · ${EQ(q)} → la tabla asume que juega; si no juega entra ${esc(q.bk)} y son ${q.fuera.toFixed(1)} pts (abajo)`).join('<br>')}</div>` : '';
  const altH = (!hist && g.alt && g.alt.length) ? g.alt.map(a => `<div class="alt"><h4>SI NO JUEGA ${esc(a.qb)} (${esc(a.eq)})</h4>
    <div class="pj p">Proyección ${f1(a.proy[0])} - ${f1(a.proy[1])} · ${esc(g.home)} gana ${Math.round(a.p_h * 100)}%</div>
    <div class="tw"><table><tr class="hd"><td></td><td>Modelo</td><td>Tu línea</td><td>Prob · justo</td></tr>${a.rows.map(r => rowH(r, false, mk)).join('')}</table></div>
    <div class="nm">Entra ${esc(a.bk)} · ${a.pen.toFixed(1)} pts para ${esc(a.eq)}</div></div>`).join('') : '';

  return `<div class="card">
    <div class="top"><span class="lg">${g.lg}</span><span>${valB}${badge}</span></div>
    <div class="fecha">${g.date}${g.hora ? ` · ${g.hora} H CENTRO` : ''}${g.week ? ` · SEMANA ${g.week}` : ''}${g.div ? '<span class="tag">DIVISIONAL</span>' : ''}</div>
    <div class="vs">${team(g.home, g.logo_h, true, g.racha && g.racha.h, g.les ? g.les.h : null)}${mid}${team(g.away, g.logo_a, false, g.racha && g.racha.a, g.les ? g.les.a : null)}</div>
    ${pr}
    ${qbH}
    <div class="mx ${abierto ? 'open' : ''}" id="m${ci}">
      <button class="mbtn" aria-expanded="${abierto}" onclick="tog(${ci})"><span>${etq}</span><span class="ar">▾</span></button>
      <div class="mbody">${edgeH}${hist ? '' : formaH(g)}<div class="tw"><table>${head}${g.rows.map(r => rowH(r, hist, mk)).join('')}</table></div>${altH}</div>
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
  ['NFL', 'FUT', 'CL', 'RES'].forEach(k => document.getElementById('t' + k).classList.toggle('on', k === tab));
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
                   racha=g.get("racha"), bajas=g.get("les"), qb_ajuste=g.get("qb"),
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
    hora_gdl = dt.datetime.now(ZoneInfo("America/Mexico_City")).strftime("%d/%m/%Y · %H:%M")
    html = HTML_PART1.replace("__ACTUALIZADO__", hora_gdl) + HTML_PART2_JS.replace("__DATA__", json.dumps(data, ensure_ascii=False, default=str))
    with open(SALIDA, "w", encoding="utf-8") as f:
        f.write(html)
    print("Listo →", SALIDA)
    descargar(SALIDA)


if __name__ == "__main__":
    main()
