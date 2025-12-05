# /Users/marcopellegrino/Desktop/Dataset Analisi Paper/run_pipeline.py
from __future__ import annotations

import sys
import math
import time
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
import yfinance as yf


# ---------------------------- CONFIG & LOGGING ---------------------------- #

@dataclass(frozen=True)
class BaseConfig:
    base_dir: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper")
    ticker_csv: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/Ticker.csv")
    out_fund_raw: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/fundamentals_raw_2009_2025.csv")
    out_fund_pivot: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/fundamentals_pivot_2009_2025.csv")
    out_fund_ticker: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/fundamentals_with_ticker_2009_2025.csv")
    out_prices: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/stock_prices_monthly_long_2009_2025.csv")
    out_indicators: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/indicators_2009_2025.csv")
    out_portfolios: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/portfolios_2009_2025.csv")
    out_backtest: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/backtest_2009_2025.csv")
    out_backtest_by_year: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/backtest_by_year_2009_2025.csv")
    out_backtest_spreads: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/backtest_spreads_2009_2025.csv")
    out_failed_tickers: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/failed_tickers_yfinance.csv")
    out_log: Path = Path("/Users/marcopellegrino/Desktop/Dataset Analisi Paper/run_pipeline.log")

    start_date: str = "2009-01-01"
    end_date: str = "2025-12-31"
    yf_interval: str = "1mo"
    yf_batch_size: int = 40
    yf_pause_sec: float = 0.5

    TAGS: Tuple[str, ...] = (
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "Revenues",
        "NetIncomeLoss",
        "OperatingIncomeLoss",
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
        "CommonStockSharesOutstanding",
        "CashAndCashEquivalentsAtCarryingValue",
        "NetCashProvidedByUsedInOperatingActivities",
        "ReturnOnAssets",
        "ReturnOnEquity",
    )


def setup_logging(log_path: Optional[Path] = None, level: int = logging.INFO) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers)


# ---------------------------- HELPERS ---------------------------- #

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def zfill_cik(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(10)
    )


# ---------------------------- SEC LOADING ---------------------------- #

def find_sec_quarter_dirs(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir non trovata: {base_dir}")
    pick = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if (name.endswith("q4") or name == "2025q3") and (p / "sub.txt").exists() and (p / "num.txt").exists():
            pick.append(p)
    pick.sort(key=lambda x: x.name)
    logging.info("Cartelle SEC: %s", [d.name for d in pick])
    if not pick:
        logging.warning("Nessuna cartella SEC trovata.")
    return pick


def _sec_dtypes_sub() -> Dict[str, str]:
    return {"adsh": "string", "cik": "string", "name": "string", "form": "string", "fy": "Int64"}


def _sec_dtypes_num() -> Dict[str, str]:
    return {"adsh": "string", "tag": "string", "version": "string", "ddate": "string", "qtrs": "Int64", "uom": "string", "value": "float64"}


def load_sec_folder(folder: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        sub = pd.read_csv(folder / "sub.txt", sep="\t", dtype=_sec_dtypes_sub(), low_memory=False)
        num = pd.read_csv(folder / "num.txt", sep="\t", dtype=_sec_dtypes_num(), low_memory=False)
        sub = sub.loc[sub["form"].str.upper() == "10-K", ["adsh", "cik", "name", "fy"]].copy()
        sub["cik"] = zfill_cik(sub["cik"])
        sub = sub.dropna(subset=["adsh", "cik", "fy"])
        num = num.loc[num["tag"].isin(BaseConfig.TAGS), ["adsh", "tag", "value"]].copy()
        num = num[num["adsh"].isin(sub["adsh"])]
        return sub, num
    except Exception:
        logging.exception("Errore caricando %s", folder)
        return pd.DataFrame(), pd.DataFrame()


def build_fundamentals(base_dir: Path, cfg: BaseConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for folder in tqdm(find_sec_quarter_dirs(base_dir), desc="SEC folders", unit="dir"):
        sub, num = load_sec_folder(folder)
        if sub.empty or num.empty:
            continue
        merged = num.merge(sub, on="adsh", how="inner")[["cik", "name", "fy", "tag", "value"]]
        merged["fy"] = merged["fy"].astype("int64", errors="ignore")
        rows.append(merged)
    if not rows:
        raise RuntimeError("Nessun dato SEC valido trovato.")
    df_raw = pd.concat(rows, ignore_index=True)
    ensure_parent(cfg.out_fund_raw); df_raw.to_csv(cfg.out_fund_raw, index=False)

    df_pivot = (
        df_raw.pivot_table(index=["cik", "fy"], columns="tag", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "EarningsPerShareDiluted" in df_pivot.columns or "EarningsPerShareBasic" in df_pivot.columns:
        df_pivot["EPS"] = df_pivot.get("EarningsPerShareDiluted").fillna(df_pivot.get("EarningsPerShareBasic"))
    ensure_parent(cfg.out_fund_pivot); df_pivot.to_csv(cfg.out_fund_pivot, index=False)
    return df_raw, df_pivot


# ---------------------------- CIK → TICKER ---------------------------- #

def load_cik_map(ticker_csv: Path) -> pd.DataFrame:
    if not ticker_csv.exists():
        raise FileNotFoundError(f"Ticker CSV non trovato: {ticker_csv}")
    df_try = pd.read_csv(ticker_csv, dtype=str)
    cols = [c.lower().strip() for c in df_try.columns]
    df_try.columns = cols
    if {"ticker", "cik"} <= set(cols):
        df = df_try[["ticker", "cik"]].copy()
    else:
        df = pd.read_csv(ticker_csv, header=None, names=["ticker", "cik"], dtype=str)
    df["cik"] = zfill_cik(df["cik"]); df["ticker"] = df["ticker"].str.upper().str.strip()
    return df.dropna(subset=["ticker", "cik"]).drop_duplicates("cik", keep="first")


def attach_tickers(df_pivot: pd.DataFrame, cfg: BaseConfig) -> pd.DataFrame:
    cmap = load_cik_map(cfg.ticker_csv)
    df = df_pivot.copy(); df["cik"] = zfill_cik(df["cik"])
    df = df.merge(cmap, on="cik", how="left")
    df = df[df["ticker"].notna()].copy()
    ensure_parent(cfg.out_fund_ticker); df.to_csv(cfg.out_fund_ticker, index=False)
    return df


# ---------------------------- PREZZI (YFINANCE) ---------------------------- #

def _multiindex_orientation(mi: pd.MultiIndex) -> Optional[bool]:
    # True: level0=field ('Adj Close', 'Close'), False: level0=ticker
    lev0 = {str(x).lower() for x in mi.get_level_values(0).unique()}
    lev1 = {str(x).lower() for x in mi.get_level_values(1).unique()}
    fields = {"adj close", "close"}
    if fields & lev0: return True
    if fields & lev1: return False
    return None


def _stack_compat(adj: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas 2.2+ non vuole 'dropna' quando usi future_stack=True.
    Qui proviamo prima con future_stack=True senza dropna; se non esiste, fallback classico con dropna=True.
    """
    try:
        long_df = adj.stack(future_stack=True).reset_index()
    except TypeError:
        # pandas < 2.1 non conosce future_stack
        long_df = adj.stack(dropna=True).reset_index()
    return long_df


def _extract_adj_or_close(df_yf: pd.DataFrame) -> pd.DataFrame:
    """Ritorna long {date,ticker,adj_close} oppure vuoto. Mai raise."""
    if df_yf is None or df_yf.empty:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])

    # Multi-ticker
    if isinstance(df_yf.columns, pd.MultiIndex):
        tmp = df_yf.dropna(axis=1, how="all")
        if tmp.empty:
            return pd.DataFrame(columns=["date", "ticker", "adj_close"])
        orient = _multiindex_orientation(tmp.columns)
        if orient is None:
            return pd.DataFrame(columns=["date", "ticker", "adj_close"])
        fields_try = ["Adj Close", "Close"]
        adj = None
        if orient:
            for f in fields_try:
                if f in tmp.columns.get_level_values(0):
                    adj = tmp.xs(f, axis=1, level=0); break
        else:
            for f in fields_try:
                if f in tmp.columns.get_level_values(1):
                    adj = tmp.xs(f, axis=1, level=1); break
        if adj is None or adj.dropna(how="all").empty:
            return pd.DataFrame(columns=["date", "ticker", "adj_close"])
        adj.index = pd.to_datetime(adj.index, errors="coerce")

        long_df = _stack_compat(adj)
        long_df.columns = ["date", "ticker", "adj_close"]
        long_df["ticker"] = long_df["ticker"].astype(str).str.upper()
        return long_df.dropna(subset=["date", "adj_close"])

    # Single-ticker
    cols_lower = {c.lower(): c for c in df_yf.columns}
    col = cols_lower.get("adj close") or cols_lower.get("close")
    if not col:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])
    out = df_yf[[col]].rename(columns={col: "adj_close"}).copy()
    out["date"] = pd.to_datetime(out.index, errors="coerce")
    tk = getattr(df_yf, "ticker", None)
    if not tk:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])
    out["ticker"] = str(tk).upper()
    return out.dropna(subset=["date", "adj_close"])[["date", "ticker", "adj_close"]]


def _split_batches(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _fallback_per_ticker(tickers: List[str], cfg: BaseConfig) -> Tuple[pd.DataFrame, Set[str]]:
    frames, failed = [], set()
    for tk in tickers:
        try:
            hist = yf.Ticker(tk).history(start=cfg.start_date, end=cfg.end_date, interval=cfg.yf_interval, auto_adjust=False)
            if hist is None or hist.empty:
                failed.add(tk); continue
            hist = hist.dropna(how="all")
            cols_lower = {c.lower(): c for c in hist.columns}
            col = cols_lower.get("adj close") or cols_lower.get("close")
            if not col:
                failed.add(tk); continue
            tmp = hist[[col]].rename(columns={col: "adj_close"}).copy()
            tmp["date"] = pd.to_datetime(tmp.index, errors="coerce")
            tmp["ticker"] = tk
            tmp = tmp.dropna(subset=["date", "adj_close"])
            if not tmp.empty:
                frames.append(tmp[["date", "ticker", "adj_close"]])
            else:
                failed.add(tk)
        except Exception:
            logging.exception("Fallback history() fallito per %s", tk)
            failed.add(tk)
        time.sleep(0.15)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", "adj_close"])), failed

def _looks_problematic_symbol(tk: str) -> bool:
    tk = tk.upper()
    bad_suffixes = ("Q",)         # spesso bankruptcy/delist
    bad_contains = (".", "-", "^")  # formati non-standard yahoo
    if any(ch in tk for ch in bad_contains):
        return True
    if tk.endswith(bad_suffixes):
        return True
    # ADR/foreign pink a volte falliscono (commenta se ti serve tenerli)
    if tk.endswith("Y") and len(tk) >= 3:
        return True
    return False

def download_prices_monthly_long(tickers: List[str], cfg: BaseConfig) -> Tuple[pd.DataFrame, Set[str]]:
    frames: List[pd.DataFrame] = []
    all_failed: Set[str] = set()

    uniq = sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})
    for batch in tqdm(_split_batches(uniq, cfg.yf_batch_size), desc="Scarico prezzi (batch)", unit="batch"):
        long_df_batch = pd.DataFrame(columns=["date", "ticker", "adj_close"])
        try:
            df_yf = yf.download(
                tickers=batch,
                start=cfg.start_date,
                end=cfg.end_date,
                interval=cfg.yf_interval,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if isinstance(df_yf, pd.Series):
                df_yf = df_yf.to_frame()
            if len(batch) == 1 and not isinstance(df_yf.columns, pd.MultiIndex):
                df_yf = df_yf.copy(); df_yf.ticker = batch[0]
            long_df_batch = _extract_adj_or_close(df_yf)
        except Exception:
            logging.exception("Errore batch %s", batch)

        tickers_in_batch = set(batch)
        if long_df_batch.empty:
            fb_df, fb_failed = _fallback_per_ticker(batch, cfg)
            if not fb_df.empty:
                long_df_batch = fb_df
            all_failed.update(fb_failed)
        else:
            got = set(long_df_batch["ticker"].unique())
            missing = list(tickers_in_batch - got)
            if missing:
                fb_df, fb_failed = _fallback_per_ticker(missing, cfg)
                if not fb_df.empty:
                    long_df_batch = pd.concat([long_df_batch, fb_df], ignore_index=True)
                all_failed.update(fb_failed)

        if not long_df_batch.empty:
            frames.append(long_df_batch)

        time.sleep(cfg.yf_pause_sec)

    if frames:
        prices = pd.concat(frames, ignore_index=True)
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices = prices.dropna(subset=["date", "adj_close", "ticker"])
        prices["ticker"] = prices["ticker"].str.upper().str.strip()
        prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
        prices["year"] = prices["date"].dt.year.astype("int64")
        prices["month"] = prices["date"].dt.month.astype("int64")
        return prices[["date", "year", "month", "ticker", "adj_close"]], all_failed

    return pd.DataFrame(columns=["date", "year", "month", "ticker", "adj_close"]), all_failed


# ---------------------------- INDICATORI / PORTAFOGLI / BACKTEST ---------------------------- #

def compute_momentum_features(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = prices.copy()
    prices["ret_12m"] = prices.groupby("ticker", observed=True)["adj_close"].pct_change(12)
    momentum_dec = prices.loc[prices["month"] == 12, ["ticker", "year", "ret_12m"]].rename(columns={"ret_12m": "Momentum"})
    price_june = prices.loc[prices["month"] == 6, ["ticker", "year", "adj_close"]].rename(columns={"adj_close": "price_june"})
    return momentum_dec, price_june


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num.astype(float) / den.astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_indicators(df_fund: pd.DataFrame, momentum_dec: pd.DataFrame, price_june: pd.DataFrame) -> pd.DataFrame:
    f = df_fund.copy()
    f["fy"] = f["fy"].astype(int)
    f = f.merge(price_june, left_on=["ticker", "fy"], right_on=["ticker", "year"], how="left").drop(columns=["year"])
    eps = f.get("EPS")
    if eps is None:
        eps = f.get("EarningsPerShareDiluted", pd.Series(dtype=float)).fillna(f.get("EarningsPerShareBasic", pd.Series(dtype=float)))
    f["PE"] = _safe_div(f.get("price_june", pd.Series(dtype=float)), eps)
    f["ROE"] = _safe_div(f.get("NetIncomeLoss", pd.Series(dtype=float)), f.get("StockholdersEquity", pd.Series(dtype=float)))
    f = f.merge(momentum_dec, left_on=["ticker", "fy"], right_on=["ticker", "year"], how="left").drop(columns=["year"])
    return f


def _safe_deciles_per_year(df: pd.DataFrame, col: str, year_col: str = "fy") -> pd.Series:
    def _decile(s: pd.Series) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.dropna().nunique() < 3:
            return pd.Series(index=s.index, data=np.nan)
        try:
            return pd.qcut(s, 10, labels=False, duplicates="drop")
        except Exception:
            r = s.rank(method="average", pct=True)
            return pd.cut(r, bins=np.linspace(0, 1, 11), labels=False, include_lowest=True)
    return df.groupby(year_col, group_keys=False)[col].apply(_decile)


def build_portfolios(df_ind: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    d = df_ind.copy()
    d["PE_clean"] = d["PE"].replace([np.inf, -np.inf], np.nan)
    d["Momentum_clean"] = d["Momentum"].replace([np.inf, -np.inf], np.nan)
    d["PE_decile"] = _safe_deciles_per_year(d, "PE_clean")
    d["MOM_decile"] = _safe_deciles_per_year(d, "Momentum_clean")
    ports = {
        "value": d[d["PE_decile"] == 0],
        "growth": d[d["PE_decile"] == 9],
        "momentum_winners": d[d["MOM_decile"] == 9],
        "momentum_losers": d[d["MOM_decile"] == 0],
    }
    long_port = []
    for name, pdf in ports.items():
        if pdf.empty: continue
        tmp = pdf[["ticker", "fy"]].copy(); tmp["portfolio"] = name
        long_port.append(tmp)
    df_ports = pd.concat(long_port, ignore_index=True) if long_port else pd.DataFrame(columns=["ticker", "fy", "portfolio"])
    return df_ports, ports

# --- Patch 3a: helper per ritorno annuale composto su mensili ---
def _compound_calendar_year_return(prices_one_year: pd.DataFrame) -> Optional[float]:
    """
    Ritorno composto sui mensili nell'anno (>= 11 osservazioni).
    Scarta casi patologici con prezzo iniziale troppo basso.
    """
    if prices_one_year is None or prices_one_year.empty:
        return None

    px = prices_one_year.sort_values("date")
    if px["adj_close"].isna().any() or len(px) < 11:
        return None

    p0 = float(px.iloc[0]["adj_close"])
    if not np.isfinite(p0) or p0 <= 0.5:  # soglia robustezza penny/artefatti
        return None

    rets = px["adj_close"].pct_change().dropna()
    if rets.empty:
        return None

    comp = float(np.prod(1.0 + rets) - 1.0)
    if not np.isfinite(comp):
        return None
    return comp

# --- Patch 3b: nuova compute_annual_return_for_portfolio (compounding + winsorize) ---
def compute_annual_return_for_portfolio(port_df: pd.DataFrame, prices_long: pd.DataFrame) -> float:
    """
    Ritorno medio equal-weight del portafoglio nel FY+1.
    - Calcolo su mensili (composto), richiede >= 11 mesi disponibili.
    - Scarta p0 < 0.5 USD.
    - Winsorize 1%–99% per contenere outlier.
    """
    if port_df.empty or prices_long.empty:
        return float("nan")

    per_name: List[float] = []
    for _, row in port_df.iterrows():
        tk = row["ticker"]
        year_target = int(row["fy"]) + 1
        px = prices_long.loc[
            (prices_long["ticker"] == tk) & (prices_long["year"] == year_target),
            ["date", "adj_close"]
        ]
        r = _compound_calendar_year_return(px)
        if r is not None:
            per_name.append(r)

    if not per_name:
        return float("nan")

    q1, q99 = np.nanpercentile(per_name, 1), np.nanpercentile(per_name, 99)
    per_name = np.clip(per_name, q1, q99)

    return float(np.nanmean(per_name))


def backtest(ports: Dict[str, pd.DataFrame], prices_long: pd.DataFrame) -> pd.DataFrame:
    rows = [{"portfolio": name, "avg_return_next_fy": compute_annual_return_for_portfolio(pdf, prices_long)} for name, pdf in ports.items()]
    return pd.DataFrame(rows)


# ---------------------------- ANALISI EXTRA ---------------------------- #

def backtest_by_year(df_ports_long: pd.DataFrame, prices_long: pd.DataFrame) -> pd.DataFrame:
    """
    Ritorni equal-weight anno per anno (FY -> rendimento su FY+1) per ciascun portafoglio.
    """
    if df_ports_long.empty or prices_long.empty:
        return pd.DataFrame(columns=["fy", "portfolio", "avg_return_next_fy"])

    out_rows = []
    for (fy, portfolio), grp in df_ports_long.groupby(["fy", "portfolio"], observed=True):
        r = compute_annual_return_for_portfolio(grp[["ticker", "fy"]], prices_long)
        out_rows.append({"fy": int(fy), "portfolio": portfolio, "avg_return_next_fy": r})
    df = pd.DataFrame(out_rows).sort_values(["fy", "portfolio"]).reset_index(drop=True)
    return df


def compute_spreads(df_backtest_by_year: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola spread annuali:
      - MOM: winners - losers
      - VALUE: value - growth
    """
    if df_backtest_by_year.empty:
        return pd.DataFrame(columns=["fy", "mom_wml", "value_vmg"])

    pivot = df_backtest_by_year.pivot(index="fy", columns="portfolio", values="avg_return_next_fy")
    pivot = pivot.rename(columns={
        "momentum_winners": "MOM_WIN",
        "momentum_losers": "MOM_LOSE",
        "value": "VALUE",
        "growth": "GROWTH",
    })
    pivot["mom_wml"] = pivot["MOM_WIN"] - pivot["MOM_LOSE"]
    pivot["value_vmg"] = pivot["VALUE"] - pivot["GROWTH"]
    pivot = pivot[["mom_wml", "value_vmg"]].reset_index()
    return pivot


# ---------------------------- PIPELINE MAIN ---------------------------- #

def run_pipeline(cfg: BaseConfig) -> dict:
    logging.info("Pipeline avviata.")
    df_raw, df_pivot = build_fundamentals(cfg.base_dir, cfg)
    logging.info("Fundamentali raw: %s pivot: %s", df_raw.shape, df_pivot.shape)

    df_fund_ticker = attach_tickers(df_pivot, cfg)
    logging.info("Fundamentali con ticker: %s", df_fund_ticker.shape)
    all_tickers = df_fund_ticker["ticker"].dropna().astype(str).str.upper().unique().tolist()

    # Filtro preventivo simboli potenzialmente problematici
    filtered = [t for t in all_tickers if not _looks_problematic_symbol(t)]
    dropped = set(all_tickers) - set(filtered)
    if dropped:
        logging.info("Filtro simboli problematici: esclusi %d tickers prima del download.", len(dropped))

    # ---------- DOWNLOAD PREZZI (UNA SOLA VOLTA!) ----------
    prices, failed = download_prices_monthly_long(filtered, cfg)
    ensure_parent(cfg.out_prices); prices.to_csv(cfg.out_prices, index=False)
    if failed:
        ensure_parent(cfg.out_failed_tickers)
        pd.DataFrame(sorted(failed), columns=["ticker"]).to_csv(cfg.out_failed_tickers, index=False)
        logging.warning("Ticker senza dati: %d (vedi %s)", len(failed), cfg.out_failed_tickers)
    logging.info("Prezzi salvati: %s", prices.shape)

    # Se nessun prezzo, genera file vuoti coerenti e termina senza errori
    if prices.empty:
        logging.warning("Nessun prezzo valido scaricato. Creo output vuoti e termino.")
        pd.DataFrame(columns=["ticker", "year", "Momentum"]).to_csv(cfg.out_indicators, index=False)
        pd.DataFrame(columns=["ticker", "fy", "portfolio"]).to_csv(cfg.out_portfolios, index=False)
        pd.DataFrame(columns=["portfolio", "avg_return_next_fy"]).to_csv(cfg.out_backtest, index=False)
        pd.DataFrame(columns=["fy", "portfolio", "avg_return_next_fy"]).to_csv(cfg.out_backtest_by_year, index=False)
        pd.DataFrame(columns=["fy", "mom_wml", "value_vmg"]).to_csv(cfg.out_backtest_spreads, index=False)
        return {
            "fund_raw": df_raw.shape, "fund_pivot": df_pivot.shape, "fund_with_ticker": df_fund_ticker.shape,
            "prices": prices.shape, "failed": len(failed)
        }

    # Features prezzi + indicatori
    momentum_dec, price_june = compute_momentum_features(prices)
    indicators = compute_indicators(df_fund_ticker, momentum_dec, price_june)
    ensure_parent(cfg.out_indicators); indicators.to_csv(cfg.out_indicators, index=False)
    logging.info("Indicatori: %s", indicators.shape)

    # Portafogli
    df_ports_long, ports_dict = build_portfolios(indicators)
    ensure_parent(cfg.out_portfolios); df_ports_long.to_csv(cfg.out_portfolios, index=False)
    logging.info("Portafogli: %s", df_ports_long.shape)

    # Backtest complessivo (media su tutti gli anni)
    bt = backtest(ports_dict, prices)
    ensure_parent(cfg.out_backtest); bt.to_csv(cfg.out_backtest, index=False)
    logging.info("Backtest:\n%s", bt.to_string(index=False))

    # Analisi EXTRA: backtest per anno + spread
    bt_year = backtest_by_year(df_ports_long, prices)
    ensure_parent(cfg.out_backtest_by_year); bt_year.to_csv(cfg.out_backtest_by_year, index=False)
    spreads = compute_spreads(bt_year)
    ensure_parent(cfg.out_backtest_spreads); spreads.to_csv(cfg.out_backtest_spreads, index=False)

    summary = {
        "fund_raw": df_raw.shape,
        "fund_pivot": df_pivot.shape,
        "fund_with_ticker": df_fund_ticker.shape,
        "prices": prices.shape,
        "indicators": indicators.shape,
        "portfolios": df_ports_long.shape,
        "backtest_rows": int(len(bt)),
        "backtest_by_year_rows": int(len(bt_year)),
        "spread_rows": int(len(spreads)),
        "failed_tickers": int(len(failed)),
    }
    return summary


if __name__ == "__main__":
    cfg = BaseConfig()
    setup_logging(cfg.out_log, level=logging.INFO)
    try:
        # riduci logging verboso interno di yfinance/urllib
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        logging.getLogger("urllib3").setLevel(logging.CRITICAL)

        summary = run_pipeline(cfg)
        print("\n=== RIEPILOGO PIPELINE ===")
        for k, v in summary.items():
            print(f"{k:>22}: {v}")
        print("==========================")
        print("Log dettagliato:", cfg.out_log)

        # Manifest (utile per tracciabilità)
        manifest = {
            "paths": {
                "fund_raw": str(cfg.out_fund_raw),
                "fund_pivot": str(cfg.out_fund_pivot),
                "fund_with_ticker": str(cfg.out_fund_ticker),
                "prices": str(cfg.out_prices),
                "indicators": str(cfg.out_indicators),
                "portfolios": str(cfg.out_portfolios),
                "backtest": str(cfg.out_backtest),
                "backtest_by_year": str(cfg.out_backtest_by_year),
                "backtest_spreads": str(cfg.out_backtest_spreads),
                "failed_tickers": str(cfg.out_failed_tickers),
                "log": str(cfg.out_log),
            },
            "shapes": summary,
        }
        with open(cfg.base_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        logging.exception("Errore fatale pipeline: %s", e)
        sys.exit(1)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# === PERCORSO BASE ===
BASE_DIR = "/Users/marcopellegrino/Desktop/Dataset Analisi Paper"

# === CARICA FILE ===
bt_year = pd.read_csv(os.path.join(BASE_DIR, "backtest_by_year_2009_2025.csv"))
bt_spreads = pd.read_csv(os.path.join(BASE_DIR, "backtest_spreads_2009_2025.csv"))

# === STILE GRAFICI ===
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# === FIGURE 1: Ritorni annuali per portafoglio ===
plt.figure()
sns.lineplot(data=bt_year, x="fy", y="avg_return_next_fy", hue="portfolio", marker="o")
plt.title("Annual Returns by Portfolio (FY+1)")
plt.ylabel("Return")
plt.xlabel("Fiscal Year")
plt.legend(title="Portfolio")
plt.tight_layout()
plt.savefig("fig1_portfolio_returns.png", dpi=300)
plt.close()

# === FIGURE 2: Boxplot ritorni annuali ===
plt.figure()
sns.boxplot(data=bt_year, x="portfolio", y="avg_return_next_fy", palette="Set2")
plt.title("Distribution of Annual Returns per Portfolio")
plt.ylabel("Return")
plt.xlabel("Portfolio")
plt.tight_layout()
plt.savefig("fig2_boxplot_returns.png", dpi=300)
plt.close()

# === FIGURE 3: Long-short spreads ===
plt.figure()
sns.lineplot(data=bt_spreads, x="fy", y="mom_wml", label="Momentum WML", marker="o", color="darkgreen")
sns.lineplot(data=bt_spreads, x="fy", y="value_vmg", label="Value VMG", marker="o", color="firebrick")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
plt.title("Long-Short Factor Spreads Over Time")
plt.ylabel("Spread Return")
plt.xlabel("Fiscal Year")
plt.legend()
plt.tight_layout()
plt.savefig("fig3_long_short_spreads.png", dpi=300)
plt.close()

# === STATISTICHE: media, std, Sharpe ===
stats = bt_year.groupby("portfolio")["avg_return_next_fy"].agg(["mean", "std"])
stats["sharpe"] = stats["mean"] / stats["std"]
print("📊 Portfolio Performance Statistics:")
print(stats.round(4))

# === T-TEST: Value vs Growth ===
val = bt_year[bt_year["portfolio"] == "value"]["avg_return_next_fy"]
gro = bt_year[bt_year["portfolio"] == "growth"]["avg_return_next_fy"]
t_val_gro = ttest_ind(val, gro, nan_policy="omit", equal_var=False)
print("\n📌 T-test: Value vs Growth")
print(f"t = {t_val_gro.statistic:.3f}, p-value = {t_val_gro.pvalue:.4f}")

# === T-TEST: Momentum Winners vs Losers ===
win = bt_year[bt_year["portfolio"] == "momentum_winners"]["avg_return_next_fy"]
los = bt_year[bt_year["portfolio"] == "momentum_losers"]["avg_return_next_fy"]
t_win_los = ttest_ind(win, los, nan_policy="omit", equal_var=False)
print("\n📌 T-test: Momentum Winners vs Losers")
print(f"t = {t_win_los.statistic:.3f}, p-value = {t_win_los.pvalue:.4f}")

from scipy.stats import mannwhitneyu
import pandas as pd
import os

BASE_DIR = "/Users/marcopellegrino/Desktop/Dataset Analisi Paper"

# Carica il dataset
df = pd.read_csv(os.path.join(BASE_DIR, "backtest_by_year_2009_2025.csv"))

# Rimuove valori NaN
value = df[df['portfolio'] == 'value']['avg_return_next_fy'].dropna()
growth = df[df['portfolio'] == 'growth']['avg_return_next_fy'].dropna()
winners = df[df['portfolio'] == 'momentum_winners']['avg_return_next_fy'].dropna()
losers = df[df['portfolio'] == 'momentum_losers']['avg_return_next_fy'].dropna()

# Wilcoxon / Mann-Whitney U tests
u_vg, p_vg = mannwhitneyu(value, growth, alternative='two-sided')
u_mom, p_mom = mannwhitneyu(winners, losers, alternative='two-sided')

print(f"Wilcoxon Test - Value vs Growth: U={u_vg:.2f}, p={p_vg:.4f}")
print(f"Wilcoxon Test - Winners vs Losers: U={u_mom:.2f}, p={p_mom:.4f}")

import pandas as pd
import os
import numpy as np

BASE_DIR = "/Users/marcopellegrino/Desktop/Dataset Analisi Paper"
df = pd.read_csv(os.path.join(BASE_DIR, "backtest_by_year_2009_2025.csv"))

# Stampa colonne per debugging
print("Colonne disponibili nel CSV:")
print(df.columns)

# Prova a trovare il nome corretto della colonna 'anno'
possible_year_columns = ['year', 'fy', 'fiscal_year', 'year_fy', 'backtest_year']
year_col = None
for col in possible_year_columns:
    if col in df.columns:
        year_col = col
        break

if year_col is None:
    raise ValueError("⚠️ Nessuna colonna valida trovata per l'anno (year/fy/etc).")

df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
df = df.dropna(subset=[year_col])

# Definisci i due sottoperiodi
periods = {
    "2009–2016": (2009, 2016),
    "2017–2024": (2017, 2024)
}

# Funzione per calcolare statistiche
def compute_stats(df_period):
    results = []
    for portfolio in df_period['portfolio'].unique():
        data = df_period[df_period['portfolio'] == portfolio]['avg_return_next_fy'].dropna()
        if len(data) >= 2:
            mean_return = data.mean()
            std_return = data.std()
            sharpe_ratio = mean_return / std_return if std_return != 0 else np.nan
            results.append({
                'Portfolio': portfolio,
                'Mean Return': mean_return,
                'Std Dev': std_return,
                'Sharpe Ratio': sharpe_ratio
            })
    return pd.DataFrame(results)

# Applica la funzione ai due periodi
for label, (start, end) in periods.items():
    print(f"\nSub-Period: {label}")
    df_period = df[(df[year_col] >= start) & (df[year_col] <= end)]
    stats = compute_stats(df_period)
    print(stats.to_string(index=False))

import pandas as pd
import os

BASE_DIR = "/Users/marcopellegrino/Desktop/Dataset Analisi Paper"

# Carica prezzi mensili
prices = pd.read_csv(os.path.join(BASE_DIR, "stock_prices_monthly_long_2009_2025.csv"),
                     parse_dates=['date'])

# Carica composizione portafogli
ports = pd.read_csv(os.path.join(BASE_DIR, "portfolios_2009_2025.csv"))

# Prepara prezzi mensili: calcola return mensile per ogni ticker
prices = prices.sort_values(['ticker', 'date'])
prices['adj_close_prev'] = prices.groupby('ticker')['adj_close'].shift(1)
prices['ret_monthly'] = prices.groupby('ticker')['adj_close'].pct_change()
prices = prices.dropna(subset=['ret_monthly'])

# Aggiungi colonna anno e mese per join
prices['year'] = prices['date'].dt.year
prices['month'] = prices['date'].dt.month

# Definisci funzione per assegnare portafogli
def get_portfolio(row, ports_df):
    fy = row['year'] - 1  # assuming portfolio formed at fiscal year t → prices in t+1
    tick = row['ticker']
    sel = ports_df[(ports_df['fy'] == fy) & (ports_df['ticker'] == tick)]
    if sel.empty:
        return None
    return sel.iloc[0]['portfolio']

prices['portfolio'] = prices.apply(lambda r: get_portfolio(r, ports), axis=1)
monthly = prices.dropna(subset=['portfolio'])

# Raggruppa per mese e calcola return equal-weight per portafoglio
monthly_ret = (monthly
               .groupby(['date', 'portfolio'])['ret_monthly']
               .mean()
               .unstack()
               .reset_index())

monthly_ret.to_csv(os.path.join(BASE_DIR, "monthly_portfolio_returns.csv"), index=False)
print("File monthly_portfolio_returns.csv creato.")

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

BASE_DIR = "/Users/marcopellegrino/Desktop/Dataset Analisi Paper"

# === LOAD DATA ===
df_ports = pd.read_csv(os.path.join(BASE_DIR, "monthly_portfolio_returns.csv"))
df_ff = pd.read_csv(os.path.join(BASE_DIR, "US_Factors.csv"))

# === CLEAN COLUMN NAMES ===
df_ports.columns = df_ports.columns.str.strip().str.lower()
df_ff.columns = df_ff.columns.str.strip().str.lower()

# === CONVERT DATE ===
df_ff['date'] = pd.to_datetime(df_ff['date'], format='%m/%Y', errors='coerce')
df_ff = df_ff.dropna(subset=['date']).sort_values('date')

# === RENAME COLUMNS ===
df_ff = df_ff.rename(columns={
    'rm-rf': 'mkt_rf',
    'risk free rate': 'rf',
    'smb': 'smb',
    'hml': 'hml',
    'wml': 'wml'
})

# === CONVERT TO NUMERIC ===
ff_columns = ['mkt_rf', 'smb', 'hml', 'wml', 'rf']
for col in ff_columns:
    df_ff[col] = pd.to_numeric(df_ff[col], errors='coerce')

# === CLEAN PORTFOLIO RETURNS ===
df_ports['date'] = pd.to_datetime(df_ports['date'])
df_ports = df_ports.sort_values('date')

# === MERGE DATA ===
df = pd.merge(df_ports, df_ff, on='date', how='inner')
print("Merge completato. Righe:", len(df))

# === CALC EXCESS RETURNS ===
portfolios = ['value', 'growth', 'momentum_winners', 'momentum_losers']
for p in portfolios:
    df[p] = pd.to_numeric(df[p], errors='coerce')
    df[f'{p}_excess'] = df[p] - df['rf']

# === FAMA-FRENCH REGRESSIONS ===
results = {}

for p in portfolios:
    y = df[f'{p}_excess']
    X = df[['mkt_rf', 'smb', 'hml']]
    X = sm.add_constant(X)

    # Rimuove righe con NaN o inf in y o X
    mask = y.notna() & np.isfinite(y) & X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    y_clean = y[mask]
    X_clean = X[mask]

    model = sm.OLS(y_clean, X_clean).fit()
    results[p] = model

    print(f"\n=== Fama-French 3-Factor Regression: {p.upper()} ===")
    print(model.summary())
import pandas as pd

# Load data
indicators_path = "/Users/marcopellegrino/Desktop/Dataset Analisi Paper/indicators_2009_2025.csv"
portfolios_path = "/Users/marcopellegrino/Desktop/Dataset Analisi Paper/portfolios_2009_2025.csv"

indicators = pd.read_csv(indicators_path)
portfolios = pd.read_csv(portfolios_path)

print("Files loaded successfully.")

# Calculate Market Cap manually
indicators['market_cap'] = indicators['price_june'] * indicators['CommonStockSharesOutstanding']

# Merge indicators with portfolio labels
data = portfolios.merge(indicators, on=['ticker', 'fy'], how='left')

# Filter for 2017–2024 period
filtered = data[(data['fy'] >= 2017) & (data['fy'] <= 2024)]

# Group by portfolio
grouped = filtered.groupby('portfolio').agg({
    'Assets': 'mean',
    'market_cap': 'mean'
}).rename(columns={
    'Assets': 'Avg_Total_Assets',
    'market_cap': 'Avg_Market_Cap'
}).reset_index()

# Optional: format and print
grouped['Avg_Total_Assets'] = grouped['Avg_Total_Assets'].round(2)
grouped['Avg_Market_Cap'] = grouped['Avg_Market_Cap'].round(2)

print("\nDescriptive Stats by Portfolio (2017–2024):")
print(grouped)
