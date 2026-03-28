import streamlit as st
import duckdb
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import anthropic
import json
import re

HOME = Path(__file__).parent
FAQS_PATH = HOME / "dynamic_faqs.json"


def _load_persisted_faqs():
    if FAQS_PATH.exists():
        try:
            return json.loads(FAQS_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_persisted_faqs(all_db_faqs):
    FAQS_PATH.write_text(json.dumps(all_db_faqs, indent=2))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; }
    h1 { color: #f0e6ff; font-size: 2rem !important; }
    h2 { color: #d4eaff; font-size: 1.3rem !important; }
    h3 { color: #b8d4b8; font-size: 1rem !important; }
    .db-card {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 24px;
        cursor: pointer;
        transition: border-color 0.2s, background 0.2s;
        text-align: center;
    }
    .db-card:hover { border-color: #c8b4f0; }
    .db-card.selected { border-color: #c8b4f0; background: #1a1428; }
    .faq-item {
        background: #141414;
        border-left: 3px solid #f0c8b4;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 10px 0;
    }
    .faq-question { color: #f0e6d0; font-weight: 600; font-size: 0.95rem; margin-bottom: 6px; }
    .faq-answer { color: #c8c8b8; font-size: 0.88rem; line-height: 1.5; }
    .metric-box {
        background: #141414;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-val { font-size: 1.6rem; font-weight: 700; color: #b4e0c8; }
    .metric-lbl { font-size: 0.78rem; color: #888880; margin-top: 2px; }
    .qa-box {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 8px;
    }
    .msg-user {
        background: #1a1428;
        border-radius: 10px 10px 2px 10px;
        padding: 10px 14px;
        margin: 8px 0 4px auto;
        max-width: 75%;
        color: #e8e0f8;
        font-size: 0.92rem;
    }
    .msg-assistant {
        background: #101820;
        border-radius: 2px 10px 10px 10px;
        padding: 10px 14px;
        margin: 4px auto 8px 0;
        max-width: 90%;
        color: #d0e8f0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .sql-block {
        background: #0a0a14;
        border: 1px solid #2a2a3a;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: monospace;
        font-size: 0.82rem;
        color: #c8d8f8;
        margin: 6px 0;
        white-space: pre-wrap;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)

# ── Map helpers ───────────────────────────────────────────────────────────────
@st.cache_data
def load_zone_centroids():
    """Load NYC taxi zone centroids from the TLC shapefile."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(str(HOME / "taxi_zones/taxi_zones.shp")).to_crs("EPSG:4326")
        gdf["lat"] = gdf.geometry.centroid.y
        gdf["lon"] = gdf.geometry.centroid.x
        return gdf[["LocationID", "zone", "borough", "lat", "lon"]].rename(columns={"LocationID": "zone_id"})
    except Exception:
        return None


def _col(df, name, fallback_idx=0):
    """Return name if it exists as a column, else df.columns[fallback_idx]."""
    return name if name in df.columns else df.columns[min(fallback_idx, len(df.columns) - 1)]


def render_result_chart(cfg, df, color):
    """Render a chart from a chartconfig dict + result DataFrame."""
    try:
        chart_type = cfg.get("type", "bar")
        title = cfg.get("title", "")
        layout_kwargs = dict(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            margin=dict(l=0, r=0, t=30, b=0), height=320,
            template="plotly_dark",
            yaxis=dict(showgrid=False),
        )

        if chart_type == "histogram":
            x = _col(df, cfg.get("x", df.columns[0]))
            fig = px.histogram(df, x=x, title=title, color_discrete_sequence=[color])
        elif chart_type == "line":
            x = _col(df, cfg.get("x", df.columns[0]))
            y_raw = cfg.get("y", df.columns[1] if len(df.columns) > 1 else df.columns[0])
            y_cols = [_col(df, c, 1) for c in (y_raw if isinstance(y_raw, list) else [y_raw])]
            fig = px.line(df, x=x, y=y_cols, title=title, color_discrete_sequence=px.colors.qualitative.Bold)
        elif chart_type == "pie":
            names = _col(df, cfg.get("names", df.columns[0]))
            values = _col(df, cfg.get("values", df.columns[1] if len(df.columns) > 1 else df.columns[0]), 1)
            fig = px.pie(df, names=names, values=values, title=title,
                         color_discrete_sequence=px.colors.qualitative.Bold)
            layout_kwargs.pop("yaxis")
        elif chart_type == "candlestick":
            xc = _col(df, cfg.get("x", "trade_date"))
            oc = _col(df, cfg.get("open", "open"), 1)
            hc = _col(df, cfg.get("high", "high"), 2)
            lc = _col(df, cfg.get("low", "low"), 3)
            cc = _col(df, cfg.get("close", "close"), 4)
            fig = go.Figure(go.Candlestick(x=df[xc], open=df[oc], high=df[hc], low=df[lc], close=df[cc]))
            layout_kwargs["xaxis_rangeslider_visible"] = False
            if title:
                layout_kwargs["title"] = title
        else:  # bar (default)
            x = _col(df, cfg.get("x", df.columns[0]))
            y = _col(df, cfg.get("y", df.columns[1] if len(df.columns) > 1 else df.columns[0]), 1)
            fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=[color])

        fig.update_layout(**layout_kwargs)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render {cfg.get('type','bar')} chart: {e}")


def render_charts_faceted(cfgs, df, color):
    """Render all charts as a faceted subplot grid."""
    try:
        n = len(cfgs)
        ncols = min(2, n)
        nrows = (n + ncols - 1) // ncols
        bold = px.colors.qualitative.Bold

        specs = []
        titles = []
        for i in range(nrows):
            row_specs = []
            for j in range(ncols):
                idx = i * ncols + j
                t = cfgs[idx].get("type", "bar") if idx < n else "bar"
                row_specs.append({"type": "pie"} if t == "pie" else {"type": "xy"})
                titles.append(cfgs[idx].get("title", "") if idx < n else "")
            specs.append(row_specs)

        # only pass subplot_titles if at least one is non-empty
        kw = {"subplot_titles": titles} if any(titles) else {}
        fig = make_subplots(rows=nrows, cols=ncols, specs=specs, **kw)

        for i, cfg in enumerate(cfgs):
            row, col = i // ncols + 1, i % ncols + 1
            t = cfg.get("type", "bar")
            try:
                if t == "bar":
                    xc = _col(df, cfg.get("x", df.columns[0]))
                    yc = _col(df, cfg.get("y", df.columns[1] if len(df.columns) > 1 else df.columns[0]), 1)
                    fig.add_trace(go.Bar(x=df[xc], y=df[yc], marker_color=color, showlegend=False), row=row, col=col)
                elif t == "line":
                    xc = _col(df, cfg.get("x", df.columns[0]))
                    y_raw = cfg.get("y", df.columns[1] if len(df.columns) > 1 else df.columns[0])
                    y_cols = [_col(df, c, 1) for c in (y_raw if isinstance(y_raw, list) else [y_raw])]
                    for k, yc in enumerate(y_cols):
                        fig.add_trace(go.Scatter(x=df[xc], y=df[yc], mode="lines", name=yc,
                                                 line=dict(color=bold[k % len(bold)]),
                                                 showlegend=len(y_cols) > 1), row=row, col=col)
                elif t == "histogram":
                    xc = _col(df, cfg.get("x", df.columns[0]))
                    fig.add_trace(go.Histogram(x=df[xc], marker_color=color, showlegend=False), row=row, col=col)
                elif t == "pie":
                    nc = _col(df, cfg.get("names", df.columns[0]))
                    vc = _col(df, cfg.get("values", df.columns[1] if len(df.columns) > 1 else df.columns[0]), 1)
                    fig.add_trace(go.Pie(labels=df[nc], values=df[vc], showlegend=False), row=row, col=col)
                elif t == "candlestick":
                    xc = _col(df, cfg.get("x", "trade_date"))
                    oc = _col(df, cfg.get("open", "open"), 1)
                    hc = _col(df, cfg.get("high", "high"), 2)
                    lc = _col(df, cfg.get("low", "low"), 3)
                    cc = _col(df, cfg.get("close", "close"), 4)
                    fig.add_trace(go.Candlestick(x=df[xc], open=df[oc], high=df[hc],
                                                  low=df[lc], close=df[cc], showlegend=False),
                                  row=row, col=col)
            except Exception as e:
                st.warning(f"Could not render {t} chart in faceted view: {e}")

        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
            height=320 * nrows,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render faceted chart: {e}")
        for cfg in cfgs:
            render_result_chart(cfg, df, color)


def render_charts(cfgs, df, color, view_key):
    """Render one or more charts; show Individual/Faceted toggle when >1."""
    if not cfgs:
        return
    if len(cfgs) == 1:
        render_result_chart(cfgs[0], df, color)
        return
    mode = st.radio("Chart view", ["Individual", "Faceted"], horizontal=True,
                    key=view_key, label_visibility="collapsed")
    if mode == "Faceted":
        render_charts_faceted(cfgs, df, color)
    else:
        for cfg in cfgs:
            render_result_chart(cfg, df, color)


def render_result_map(cfg, df, color):
    """Render a plotly scatter map from a mapconfig dict + result DataFrame."""
    map_type = cfg.get("type", "scatter")

    if map_type == "zone_scatter":
        centroids = load_zone_centroids()
        if centroids is None:
            st.warning("Zone shapefile not found — cannot render map.")
            return
        zone_col = cfg.get("zone_col", "PULocationID")
        if zone_col not in df.columns:
            st.warning(f"Column '{zone_col}' not in results.")
            return
        df = df.merge(centroids, left_on=zone_col, right_on="zone_id", how="left").dropna(subset=["lat", "lon"])
        if "label_col" not in cfg:
            cfg["label_col"] = "zone"

    lat_col   = cfg.get("lat_col", "lat")
    lon_col   = cfg.get("lon_col", "lon")
    size_col  = cfg.get("size_col")
    color_col = cfg.get("color_col", size_col)
    label_col = cfg.get("label_col")

    if lat_col not in df.columns or lon_col not in df.columns:
        st.warning("Map requested but lat/lon columns not found in result.")
        return

    fig = px.scatter_mapbox(
        df,
        lat=lat_col, lon=lon_col,
        size=size_col if size_col and size_col in df.columns else None,
        color=color_col if color_col and color_col in df.columns else None,
        hover_name=label_col if label_col and label_col in df.columns else None,
        color_continuous_scale="Viridis",
        mapbox_style="carto-darkmatter",
        zoom=10,
        center={"lat": 40.75, "lon": -73.97},
        template="plotly_dark",
        size_max=40,
    )
    fig.update_layout(
        paper_bgcolor="#0a0a0a",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        coloraxis_colorbar=dict(title=color_col or ""),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Database registry ─────────────────────────────────────────────────────────
DATABASES = {
    "nyc_taxi": {
        "label": "NYC Yellow Taxi",
        "icon": "🚕",
        "description": "500K rides from NYC Jan 2025",
        "type": "duckdb_parquet",
        "path": str(HOME / "taxi_sample.parquet"),
        "color": "#f0c8a0",
    },
    "stocks": {
        "label": "Stock Prices",
        "icon": "📈",
        "description": "Daily OHLCV data for major tickers",
        "type": "duckdb",
        "path": str(HOME / "test.db"),
        "color": "#a0d4b8",
    },
    "ecommerce": {
        "label": "E-Commerce Clickstream",
        "icon": "🛒",
        "description": "165K online shopping sessions",
        "type": "duckdb",
        "path": str(HOME / "test.db"),
        "color": "#c8b4f0",
    },
}

def _parquet_expr(path):
    """Return a DuckDB-ready read_parquet(...) argument for a path or list of paths."""
    if isinstance(path, list):
        files = ", ".join(f"'{p}'" for p in path)
        return f"[{files}]"
    return f"'{path}'"

# ── Queries per database ──────────────────────────────────────────────────────
def load_main_metric(db_key):
    db = DATABASES[db_key]

    if db_key == "nyc_taxi":
        con = duckdb.connect()
        F = _parquet_expr(db["path"])
        df = con.execute(f"""
            SELECT tpep_pickup_datetime::DATE AS date, COUNT(*) AS trips
            FROM read_parquet({F})
            WHERE tpep_pickup_datetime >= '2025-01-01' AND tpep_pickup_datetime < '2026-01-01'
            GROUP BY date ORDER BY date
        """).df()
        df.columns = ["date", "value"]
        return df, "Trips per Day", "Trips"

    elif db_key == "stocks":
        con = _sqlite_to_duckdb(db["path"])
        df = con.execute("""
            SELECT trade_date AS date, ticker,
                   ROUND(AVG(close), 2) AS avg_close
            FROM stock_prices
            GROUP BY trade_date, ticker
            ORDER BY trade_date
        """).df()
        return df, "Avg Closing Price by Day", "Close Price (USD)"

    elif db_key == "ecommerce":
        con = _sqlite_to_duckdb(db["path"])
        df = con.execute("""
            SELECT
                make_date(year, month, day) AS date,
                COUNT(DISTINCT session_id) AS sessions,
                COUNT(*) AS pageviews,
                ROUND(AVG(price), 2) AS avg_price
            FROM ecommerce_clickstream
            GROUP BY date ORDER BY date
        """).df()
        return df, "Sessions per Day", "Sessions"


def load_faq_data(db_key):
    db = DATABASES[db_key]

    if db_key == "nyc_taxi":
        con = duckdb.connect()
        F = _parquet_expr(db["path"])

        busiest = con.execute(f"""
            SELECT tpep_pickup_datetime::DATE AS d, COUNT(*) AS n
            FROM read_parquet({F}) GROUP BY d ORDER BY n DESC LIMIT 1
        """).df()

        payment = con.execute(f"""
            SELECT CASE payment_type WHEN 1 THEN 'Credit Card' WHEN 2 THEN 'Cash'
                   ELSE 'Other' END AS method,
                   COUNT(*) AS trips, ROUND(AVG(tip_amount),2) AS avg_tip
            FROM read_parquet({F})
            WHERE payment_type IN (1,2)
            GROUP BY payment_type ORDER BY trips DESC
        """).df()

        airport = con.execute(f"""
            SELECT ROUND(AVG(total_amount),2) AS avg_fare,
                   ROUND(AVG(trip_distance),2) AS avg_miles,
                   COUNT(*) AS trips
            FROM read_parquet({F})
            WHERE PULocationID IN (132,138) OR DOLocationID IN (132,138)
        """).df()

        peak_hour = con.execute(f"""
            SELECT EXTRACT(hour FROM tpep_pickup_datetime)::INT AS hour,
                   COUNT(*) AS trips
            FROM read_parquet({F})
            GROUP BY hour ORDER BY trips DESC LIMIT 1
        """).df()

        return [
            {
                "q": "What was the single busiest day of 2025?",
                "a": f"{busiest['d'].iloc[0]} with {busiest['n'].iloc[0]:,} trips.",
            },
            {
                "q": "Do credit card riders tip more than cash riders?",
                "a": (f"Yes. Credit card riders averaged ${payment[payment['method']=='Credit Card']['avg_tip'].iloc[0]:.2f} tip, "
                      f"while cash riders recorded $0.00 (tips not captured electronically)."),
            },
            {
                "q": "What does an average airport trip look like?",
                "a": (f"{airport['avg_miles'].iloc[0]:.1f} miles, ${airport['avg_fare'].iloc[0]:.2f} total. "
                      f"There were {airport['trips'].iloc[0]:,} airport trips in 2025."),
            },
            {
                "q": "What hour of the day sees the most pickups?",
                "a": f"{int(peak_hour['hour'].iloc[0])}:00 — {peak_hour['trips'].iloc[0]:,} total pickups at that hour across the year.",
            },
            {
                "q": "How much does a typical ride cost?",
                "a": "The median fare is around $18–20. Congestion surcharge adds $2.50 in Manhattan, and JFK flat rate is $70.",
            },
            {
                "q": "Which neighborhoods generate the most rides?",
                "a": "Upper East Side (zones 236/237) and Midtown (zones 161/162) dominate pickup volume. JFK is the busiest single-location pickup zone.",
            },
        ]

    elif db_key == "stocks":
        con = _sqlite_to_duckdb(db["path"])

        tickers = con.execute("SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker").df()
        ticker_list = ", ".join(tickers["ticker"].tolist())

        best = con.execute("""
            SELECT ticker,
                   ROUND((MAX(close)-MIN(close))*100.0/MIN(close),1) AS pct_gain
            FROM stock_prices GROUP BY ticker ORDER BY pct_gain DESC LIMIT 1
        """).df()

        vol = con.execute("""
            SELECT ticker, ROUND(AVG(volume)/1e6,1) AS avg_vol_m
            FROM stock_prices GROUP BY ticker ORDER BY avg_vol_m DESC LIMIT 1
        """).df()

        spread = con.execute("""
            SELECT ticker, ROUND(AVG(high-low),2) AS avg_spread
            FROM stock_prices GROUP BY ticker ORDER BY avg_spread DESC LIMIT 1
        """).df()

        return [
            {
                "q": "Which tickers are in the database?",
                "a": f"{ticker_list}.",
            },
            {
                "q": "Which stock gained the most over the period?",
                "a": f"{best['ticker'].iloc[0]} with a {best['pct_gain'].iloc[0]:.1f}% price gain from low to high.",
            },
            {
                "q": "Which stock is most actively traded by volume?",
                "a": f"{vol['ticker'].iloc[0]} averages {vol['avg_vol_m'].iloc[0]:.1f}M shares per day.",
            },
            {
                "q": "Which stock has the widest daily price swings?",
                "a": f"{spread['ticker'].iloc[0]} has the largest avg daily high–low spread of ${spread['avg_spread'].iloc[0]:.2f}.",
            },
            {
                "q": "How reliable is closing price as a signal?",
                "a": "Closing price is the most-cited reference but can be influenced by end-of-day order flow. VWAP is often a better intraday benchmark.",
            },
            {
                "q": "What's the best way to detect a breakout?",
                "a": "Look for a close above the 20-day high on above-average volume — a classic momentum breakout signal.",
            },
        ]

    elif db_key == "ecommerce":
        con = _sqlite_to_duckdb(db["path"])

        top_cat = con.execute("""
            SELECT page1_main_category AS cat, COUNT(*) AS views
            FROM ecommerce_clickstream GROUP BY cat ORDER BY views DESC LIMIT 1
        """).df()

        avg_pages = con.execute("""
            SELECT ROUND(AVG(c),1) AS avg_pages
            FROM (SELECT session_id, COUNT(*) AS c FROM ecommerce_clickstream GROUP BY session_id)
        """).df()

        countries = con.execute("""
            SELECT COUNT(DISTINCT country) AS n FROM ecommerce_clickstream
        """).df()

        avg_price = con.execute("""
            SELECT ROUND(AVG(price),2) AS avg FROM ecommerce_clickstream
        """).df()

        return [
            {
                "q": "What is the most viewed product category?",
                "a": f"Category {top_cat['cat'].iloc[0]} (encoded ID) has the highest page views with {top_cat['views'].iloc[0]:,} hits.",
            },
            {
                "q": "How many pages does a typical session visit?",
                "a": f"On average {avg_pages['avg_pages'].iloc[0]:.1f} pages per session — suggesting moderate browsing depth before a decision.",
            },
            {
                "q": "How many countries are represented?",
                "a": f"{countries['n'].iloc[0]} distinct countries appear in the data.",
            },
            {
                "q": "What is the average product price?",
                "a": f"${avg_price['avg'].iloc[0]:.2f} average across all viewed items.",
            },
            {
                "q": "Does colour or location influence purchases?",
                "a": "Yes — the dataset includes colour and location attributes per page view, enabling A/B-style segmentation by these dimensions.",
            },
            {
                "q": "Can I measure conversion rate?",
                "a": "The clickstream tracks page sequences per session. Comparing sessions that reached a final purchase page vs. those that dropped off gives a proxy conversion rate.",
            },
        ]


# ── Schema introspection ──────────────────────────────────────────────────────
def get_schema_description(db_key):
    db = DATABASES[db_key]
    if db_key == "nyc_taxi":
        return """Table: trips  (queried as read_parquet('taxi_sample.parquet'))
Columns:
  VendorID             INTEGER   -- 1=Creative Mobile, 2=VeriFone
  tpep_pickup_datetime TIMESTAMP -- pickup timestamp
  tpep_dropoff_datetime TIMESTAMP -- dropoff timestamp
  passenger_count      DOUBLE    -- number of passengers
  trip_distance        DOUBLE    -- miles
  RatecodeID           DOUBLE    -- 1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group
  store_and_fwd_flag   VARCHAR
  PULocationID         INTEGER   -- pickup taxi zone (1-263)
  DOLocationID         INTEGER   -- dropoff taxi zone (1-263)
  payment_type         BIGINT    -- 1=Credit Card, 2=Cash, 3=No Charge, 4=Dispute
  fare_amount          DOUBLE
  extra                DOUBLE
  mta_tax              DOUBLE
  tip_amount           DOUBLE
  tolls_amount         DOUBLE
  improvement_surcharge DOUBLE
  total_amount         DOUBLE
  congestion_surcharge DOUBLE
  airport_fee          DOUBLE
  cbd_congestion_fee   DOUBLE
Date range: 2025-01-01 to 2025-01-31. ~500K rows total.
Engine: DuckDB. Use read_parquet('taxi_sample.parquet') in FROM clause."""

    con = _sqlite_to_duckdb(db["path"])
    if db_key == "stocks":
        sample = con.execute("SELECT * FROM stock_prices LIMIT 3").fetchall()
        tickers = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM stock_prices").fetchall()]
        return f"""Table: stock_prices
Columns: id INTEGER, ticker VARCHAR, trade_date VARCHAR (YYYY-MM-DD), open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume INTEGER
Tickers available: {', '.join(tickers)}
Sample rows: {sample}
Engine: DuckDB. Use standard SQL."""

    elif db_key == "ecommerce":
        sample = con.execute("SELECT * FROM ecommerce_clickstream LIMIT 2").fetchall()
        return f"""Table: ecommerce_clickstream
Columns:
  year INTEGER, month INTEGER, day INTEGER  -- build date with make_date(year, month, day)
  order_seq INTEGER, country INTEGER, session_id INTEGER
  page1_main_category INTEGER  -- product category code
  page2_clothing_model VARCHAR -- clothing model code (e.g. 'A13')
  colour INTEGER, location INTEGER, model_photography INTEGER
  price DOUBLE, price2 DOUBLE  -- item price
  page INTEGER                 -- page number in session
Engine: DuckDB. Use make_date(year, month, day) to build dates.
Sample rows: {sample}"""


def _sqlite_to_duckdb(sqlite_path):
    """Load all tables from a SQLite database into a DuckDB in-memory connection."""
    conn_sq = sqlite3.connect(sqlite_path)
    tables = [r[0] for r in conn_sq.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    con = duckdb.connect()
    for table in tables:
        df_t = pd.read_sql(f"SELECT * FROM \"{table}\"", conn_sq)
        con.register(table, df_t)
    conn_sq.close()
    return con


def run_sql(db_key, sql):
    """Execute SQL and return a DataFrame."""
    db = DATABASES[db_key]
    if db_key == "nyc_taxi":
        con = duckdb.connect()
        return con.execute(sql).df()
    else:
        con = _sqlite_to_duckdb(db["path"])
        return con.execute(sql).df()


def ask_claude(db_key, question, api_key, history):
    """Stream Claude's response. Yields (type, content) tuples:
       ('sql', sql_string) | ('text', chunk) | ('df', dataframe) | ('error', msg)
    """
    schema = get_schema_description(db_key)
    db = DATABASES[db_key]

    system = f"""You are a data analyst assistant for the {db['label']} dataset.
When the user asks a question, you MUST:
1. Write a SQL query that answers it.
2. Return it in a ```sql ... ``` fenced block — NOTHING before the fence.
3. After the fence, give a concise 1-3 sentence explanation of what the query does / what to expect.
Keep queries efficient and add LIMIT 500 unless the user asks for everything.

CHARTS: After your explanation, include one or more ```chartconfig ... ``` blocks (one per chart). You may include multiple when different views genuinely add value (e.g. a bar chart of totals AND a line chart of the trend). Each block is a JSON object. Use exact column names from the SQL result. Supported types:
- "bar"          → {{\"type\":\"bar\",\"x\":\"col\",\"y\":\"col\",\"title\":\"...\"}}
- "line"         → {{\"type\":\"line\",\"x\":\"col\",\"y\":\"col_or_list\",\"title\":\"...\"}}
- "histogram"    → {{\"type\":\"histogram\",\"x\":\"col\",\"title\":\"...\"}}
- "pie"          → {{\"type\":\"pie\",\"names\":\"col\",\"values\":\"col\",\"title\":\"...\"}}
- "candlestick"  → {{\"type\":\"candlestick\",\"x\":\"date_col\",\"open\":\"open\",\"high\":\"high\",\"low\":\"low\",\"close\":\"close\",\"title\":\"...\"}}
Guidelines: candlestick for OHLC data; pie for distributions ≤10 categories; histogram for single numeric columns; line for time series; bar otherwise.
Always include at least one chartconfig block unless the result is a single scalar value.

MAPS: If the user asks for a map, or if the result contains geographic data, include a ```mapconfig ... ``` block after your explanation with a JSON object. Supported types:
- "zone_scatter": for NYC taxi zone IDs → automatically joined to lat/lon centroids.
  Example: ```mapconfig
{{"type":"zone_scatter","zone_col":"PULocationID","size_col":"trips","color_col":"trips"}}```
- "scatter": for results that already have lat and lon columns.
  Example: ```mapconfig
{{"type":"scatter","lat_col":"lat","lon_col":"lon","size_col":"count","color_col":"count","label_col":"name"}}```
Only include a mapconfig block when geographic display genuinely adds value.

DATABASE SCHEMA:
{schema}"""

    messages = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    client = anthropic.Anthropic(api_key=api_key)
    full_text = ""

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            full_text += chunk
            yield ("text", chunk)

    # Extract SQL block
    match = re.search(r"```sql\s*(.*?)```", full_text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        yield ("sql", sql)
        try:
            df = run_sql(db_key, sql)
            yield ("df", df)
        except Exception as e:
            yield ("error", str(e))

    # Extract all chart config blocks
    chart_cfgs = []
    for m in re.findall(r"```chartconfig\s*(.*?)```", full_text, re.DOTALL | re.IGNORECASE):
        try:
            chart_cfgs.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            pass
    if chart_cfgs:
        yield ("chart_configs", chart_cfgs)

    # Extract optional map config block
    map_match = re.search(r"```mapconfig\s*(.*?)```", full_text, re.DOTALL | re.IGNORECASE)
    if map_match:
        try:
            yield ("map_config", json.loads(map_match.group(1).strip()))
        except json.JSONDecodeError:
            pass

    yield ("done", full_text)


# ── UI ────────────────────────────────────────────────────────────────────────
# ── Sidebar: API key ──────────────────────────────────────────────────────────
import os
_env_key = os.environ.get("ANTHROPIC_API_KEY", "")
with st.sidebar:
    if _env_key:
        api_key_input = _env_key
    else:
        st.markdown("### 🔑 Claude API Key")
        api_key_input = st.text_input(
            "Anthropic API key",
            type="password",
            placeholder="sk-ant-...",
            label_visibility="collapsed",
        )
        if not api_key_input:
            st.warning("Enter key to enable Q&A")
    st.markdown("---")
    st.markdown("**Q&A** uses Claude to convert your question to SQL, runs it, and shows the results.")

st.markdown("## 📊 Data Explorer")
st.markdown("<p style='color:#888880;margin-top:-12px'>Select a dataset to explore</p>", unsafe_allow_html=True)

# Database selector
if "selected_db" not in st.session_state:
    st.session_state.selected_db = None

cols = st.columns(len(DATABASES))
for col, (key, db) in zip(cols, DATABASES.items()):
    with col:
        selected = st.session_state.selected_db == key
        border = "#c8b4f0" if selected else "#2a2a2a"
        bg = "#1a1428" if selected else "#141414"
        if st.button(
            f"{db['icon']}  {db['label']}\n\n_{db['description']}_",
            key=f"btn_{key}",
            use_container_width=True,
        ):
            st.session_state.selected_db = key
            st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
if st.session_state.selected_db:
    key = st.session_state.selected_db
    db  = DATABASES[key]

    st.markdown("---")
    st.markdown(f"### {db['icon']} {db['label']}")

    with st.spinner("Loading data..."):
        df, chart_title, y_label = load_main_metric(key)

    # ── Chart ─────────────────────────────────────────────────────────────────
    if key == "stocks":
        fig = px.line(df, x="date", y="avg_close", color="ticker",
                      title=chart_title,
                      labels={"date": "Date", "avg_close": y_label, "ticker": "Ticker"},
                      template="plotly_dark",
                      color_discrete_sequence=px.colors.qualitative.Bold)
    elif key == "ecommerce":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["sessions"],
            mode="lines", name="Sessions",
            line=dict(color=db["color"], width=2),
            fill="tozeroy", fillcolor="rgba(245,66,167,0.1)"
        ))
        fig.update_layout(title=chart_title,
                          xaxis_title="Date", yaxis_title=y_label,
                          template="plotly_dark",
                          yaxis=dict(showgrid=False))
    else:  # taxi
        # 7-day rolling average
        df["rolling"] = df["value"].rolling(7, center=True).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["value"],
            mode="lines", name="Daily trips",
            line=dict(color="#3a3a3a", width=10),
            opacity=1
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rolling"],
            mode="lines", name="7-day avg",
            line=dict(color=db["color"], width=2.5)
        ))
        fig.update_layout(title=chart_title,
                          xaxis_title="Date", yaxis_title=y_label,
                          template="plotly_dark", legend=dict(orientation="h"),
                          yaxis=dict(showgrid=False))

    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        margin=dict(l=0, r=0, t=40, b=0),
        height=380,
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    if key == "nyc_taxi":
        total = df["value"].sum()
        avg   = df["value"].mean()
        peak  = df["value"].max()
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="metric-box"><div class="metric-val">{total/1e6:.1f}M</div><div class="metric-lbl">Total trips (2025)</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-box"><div class="metric-val">{avg:,.0f}</div><div class="metric-lbl">Avg trips per day</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-box"><div class="metric-val">{peak:,.0f}</div><div class="metric-lbl">Peak single day</div></div>', unsafe_allow_html=True)

    elif key == "stocks":
        n_tickers = df["ticker"].nunique()
        days = df["date"].nunique()
        price_range = f"${df['avg_close'].min():.0f} – ${df['avg_close'].max():.0f}"
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="metric-box"><div class="metric-val">{n_tickers}</div><div class="metric-lbl">Tickers tracked</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-box"><div class="metric-val">{days}</div><div class="metric-lbl">Trading days</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-box"><div class="metric-val">{price_range}</div><div class="metric-lbl">Close price range</div></div>', unsafe_allow_html=True)

    elif key == "ecommerce":
        total_sess = df["sessions"].sum()
        total_pv   = df["pageviews"].sum() if "pageviews" in df else 0
        avg_price  = df["avg_price"].mean() if "avg_price" in df else 0
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="metric-box"><div class="metric-val">{total_sess:,}</div><div class="metric-lbl">Total sessions</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-box"><div class="metric-val">{total_pv:,}</div><div class="metric-lbl">Total page views</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-box"><div class="metric-val">${avg_price:.2f}</div><div class="metric-lbl">Avg item price</div></div>', unsafe_allow_html=True)

    # ── Q&A section ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🤖 Ask anything about this dataset")

    # Per-database chat history and dynamic FAQs stored in session state
    history_key  = f"chat_history_{key}"
    dyn_faq_key  = f"dynamic_faqs_{key}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    if dyn_faq_key not in st.session_state:
        st.session_state[dyn_faq_key] = _load_persisted_faqs().get(key, [])

    # Render existing conversation
    for t_idx, turn in enumerate(st.session_state[history_key]):
        if turn["role"] == "user":
            st.markdown(f'<div class="msg-user">🧑 {turn["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-assistant">{turn["display"]}</div>', unsafe_allow_html=True)
            if turn.get("sql"):
                st.markdown(f'<div class="sql-block">{turn["sql"]}</div>', unsafe_allow_html=True)
            if turn.get("df") is not None and len(turn["df"]) > 0:
                df_r = turn["df"]
                st.dataframe(df_r, use_container_width=True, height=min(300, 40 + 35 * len(df_r)))
                if turn.get("map_cfg"):
                    render_result_map(turn["map_cfg"], df_r, db["color"])
                if turn.get("chart_cfgs"):
                    render_charts(turn["chart_cfgs"], df_r, db["color"], f"chart_view_{key}_{t_idx}")
                elif not turn.get("map_cfg") and len(df_r.columns) == 2 and pd.api.types.is_numeric_dtype(df_r.iloc[:, 1]):
                    render_result_chart({"type": "bar", "x": df_r.columns[0], "y": df_r.columns[1]}, df_r, db["color"])

    # Input row
    col_input, col_btn, col_clear = st.columns([8, 1, 1])
    with col_input:
        question = st.text_input(
            "question", label_visibility="collapsed",
            placeholder="e.g. What are the top 10 busiest pickup zones?",
            key=f"q_input_{key}",
        )
    with col_btn:
        ask = st.button("Ask", use_container_width=True, type="primary")
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state[history_key] = []
            st.rerun()

    if ask and question.strip():
        if not api_key_input:
            st.warning("⚠️ Open the sidebar (top-left ›) and enter your Anthropic API key.")
        else:
            st.markdown(f'<div class="msg-user">🧑 {question}</div>', unsafe_allow_html=True)
            placeholder = st.empty()
            streamed_text = ""
            final_sql = None
            result_df = None
            error_msg = None
            map_cfg = None
            chart_cfgs = []

            def _clean(text):
                """Strip sql, chartconfig, and mapconfig fences from display text."""
                text = re.sub(r"```sql.*?```", "", text, flags=re.DOTALL)
                text = re.sub(r"```chartconfig.*?```", "", text, flags=re.DOTALL)
                text = re.sub(r"```mapconfig.*?```", "", text, flags=re.DOTALL)
                return text.strip()

            with st.spinner("Thinking…"):
                for event_type, content in ask_claude(
                    key, question, api_key_input,
                    st.session_state[history_key]
                ):
                    if event_type == "text":
                        streamed_text += content
                        placeholder.markdown(
                            f'<div class="msg-assistant">🤖 {_clean(streamed_text)}</div>',
                            unsafe_allow_html=True
                        )
                    elif event_type == "sql":
                        final_sql = content
                        st.markdown(f'<div class="sql-block">{final_sql}</div>', unsafe_allow_html=True)
                    elif event_type == "df":
                        result_df = content
                    elif event_type == "chart_configs":
                        chart_cfgs = content
                    elif event_type == "map_config":
                        map_cfg = content
                    elif event_type == "error":
                        error_msg = content

            display_text = _clean(streamed_text)
            placeholder.markdown(
                f'<div class="msg-assistant">🤖 {display_text}</div>',
                unsafe_allow_html=True
            )

            if result_df is not None and len(result_df) > 0:
                st.dataframe(result_df, use_container_width=True,
                             height=min(300, 40 + 35 * len(result_df)))
                if map_cfg:
                    render_result_map(map_cfg, result_df, db["color"])
                if chart_cfgs:
                    render_charts(chart_cfgs, result_df, db["color"], f"chart_view_{key}_new")
                elif not map_cfg and len(result_df.columns) == 2 and pd.api.types.is_numeric_dtype(result_df.iloc[:, 1]):
                    render_result_chart({"type": "bar", "x": result_df.columns[0], "y": result_df.columns[1]}, result_df, db["color"])

                # Add successful answer to dynamic FAQ
                parts = []
                if display_text:
                    parts.append(display_text)
                # Append a plain-text summary of the top results
                preview = result_df.head(5)
                rows_txt = "; ".join(
                    " | ".join(f"{col}: {val}" for col, val in row.items())
                    for _, row in preview.iterrows()
                )
                total = len(result_df)
                parts.append(f"Top result{'s' if total > 1 else ''} ({total:,} row{'s' if total != 1 else ''} total): {rows_txt}")
                st.session_state[dyn_faq_key].append({"q": question, "a": "  ".join(parts)})
                persisted = _load_persisted_faqs()
                persisted[key] = st.session_state[dyn_faq_key]
                _save_persisted_faqs(persisted)
            elif error_msg:
                st.error(f"SQL error: {error_msg}")

            # Save to history
            st.session_state[history_key].append({"role": "user", "content": question})
            st.session_state[history_key].append({
                "role": "assistant",
                "content": streamed_text,
                "display": display_text,
                "sql": final_sql,
                "df": result_df,
                "map_cfg": map_cfg,
                "chart_cfgs": chart_cfgs,
            })

    # ── FAQ ───────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### 💬 Key questions this dataset can answer")
    with st.spinner("Loading insights..."):
        faqs = load_faq_data(key)
    all_faqs = faqs + st.session_state.get(dyn_faq_key, [])
    for faq in all_faqs:
        st.markdown(f"""
        <div class="faq-item">
          <div class="faq-question">Q: {faq['q']}</div>
          <div class="faq-answer">{faq['a']}</div>
        </div>""", unsafe_allow_html=True)

else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888880;text-align:center;font-size:1.1rem'>👆 Select a dataset above to get started</p>", unsafe_allow_html=True)
