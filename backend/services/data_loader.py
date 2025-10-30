from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Dict

from ..core.config import DATA_DIR, CACHE_DIR
from .preprocessing import build_spot_table, build_timeseries, join_spots_timeseries
from .metrics import build_ml_edges


def load_excel_sources() -> Dict[str, pd.DataFrame]:
    sources = {}
    candidates = [
        "3_관광지분석_삼성카드.xlsx",
        "3_관광지분석_LG유플러스.xlsx",
        "1_생활인구분석_LG유플러스.xlsx",
        "1_생활인구분석_경기데이터드림.xlsx",
        "5_식품위생업소소비분석_삼성카드.xlsx",
        "5_식품위생업소소비분석_경기데이터드림.xlsx",
        "4_관광지트렌드분석_LG유플러스.xlsx",
        "4_관광지트렌드분석_삼성카드.xlsx",
        "6_소비경제규모추정.xlsx",
    ]
    for name in candidates:
        path = DATA_DIR / name
        if not path.exists():
            continue
        try:
            # 모든 시트를 순회해서 수집
            xl = pd.ExcelFile(path, engine="openpyxl")
            for sheet in xl.sheet_names:
                try:
                    df = xl.parse(sheet)
                    if not df.empty:
                        sources[f"{name}::{sheet}"] = df
                except Exception:
                    continue
        except Exception:
            # 엔진 문제 등으로 실패 시 기본 read_excel 한번 더 시도
            try:
                df = pd.read_excel(path)
                if not df.empty:
                    sources[name] = df
            except Exception:
                continue
    return sources


def save_parquet(df: pd.DataFrame, name: str) -> Path:
    path = CACHE_DIR / name
    df.to_parquet(path, index=False)
    return path


def build_caches_from_excels() -> Dict[str, Path]:
    sources = load_excel_sources()
    spots_list = []
    ts_list = []
    for df in sources.values():
        spots = build_spot_table(df)
        ts = build_timeseries(df)
        if not spots.empty:
            spots_list.append(spots)
        if not ts.empty:
            ts_list.append(ts)
    if len(spots_list) == 0 or len(ts_list) == 0:
        return {}
    spots_all = pd.concat(spots_list, ignore_index=True).drop_duplicates(subset=["spot_id"]) 
    ts_all = pd.concat(ts_list, ignore_index=True)
    spots_all, ts_all = join_spots_timeseries(spots_all, ts_all)

    p_spots = save_parquet(spots_all, "spots.parquet")
    p_ts = save_parquet(ts_all, "timeseries.parquet")
    # ML 기반 엣지도 즉시 구축
    build_ml_edges(p_ts)
    return {"spots": p_spots, "timeseries": p_ts}


