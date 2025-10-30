from __future__ import annotations

import itertools
import pandas as pd
from pathlib import Path
from ..core.config import CACHE_DIR
import numpy as np
from sklearn.cluster import KMeans


def compute_concurrency(ts: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
    # ts: [time, spot_id, value]
    pivot = ts.pivot_table(index="time", columns="spot_id", values="value", aggfunc="mean")
    pivot = pivot.fillna(0.0)
    cm = pivot.corr()
    cm.index.name = "source"
    cm.columns.name = "target"
    corr = cm.stack().rename("C").reset_index()
    corr = corr[corr["source"] < corr["target"]]  # 대각/중복 제거
    corr = corr.sort_values("C", ascending=False).head(top_k)
    return corr


def compute_sequential_strength(ts: pd.DataFrame, delta_hours: int = 1, top_k: int = 100) -> pd.DataFrame:
    # 간단한 시차 상관 기반
    lead = ts.copy()
    lead["time"] = lead["time"] + pd.to_timedelta(delta_hours, unit="h")
    merged = ts.merge(lead, on=["time"], how="inner", suffixes=("_i", "_j"))
    # 같은 spot 간 연결은 제외
    merged = merged[merged["spot_id_i"] != merged["spot_id_j"]]

    # 시간별로 평균 제거 후 공분산/분산 ratio 근사
    def ratio(g: pd.DataFrame) -> float:
        vi = g["value_i"] - g["value_i"].mean()
        vj = g["value_j"] - g["value_j"].mean()
        denom = (vi.std() * vj.std())
        if denom == 0 or pd.isna(denom):
            return 0.0
        return float((vi * vj).mean() / denom)

    grouped = merged.groupby(["spot_id_i", "spot_id_j"]).apply(ratio).reset_index(name="S")
    grouped = grouped.rename(columns={"spot_id_i": "source", "spot_id_j": "target"})
    grouped = grouped.sort_values("S", ascending=False).head(top_k)
    return grouped


def combine_scores(c_df: pd.DataFrame, s_df: pd.DataFrame, w_c: float = 0.5, w_s: float = 0.5) -> pd.DataFrame:
    df = c_df.merge(s_df, on=["source", "target"], how="outer").fillna(0.0)
    df["A"] = w_c * df.get("C", 0.0) + w_s * df.get("S", 0.0)
    return df


def build_edges_from_timeseries(ts_path: Path | None = None, limit: int = 100) -> pd.DataFrame:
    if ts_path is None:
        ts_path = CACHE_DIR / "timeseries.parquet"
    if not ts_path.exists():
        return pd.DataFrame(columns=["source", "target", "C", "S", "A"]) 
    try:
        ts = pd.read_parquet(ts_path)
    except Exception:
        # 손상된 파케이로 추정 → 삭제하고 빈 결과 반환(상위 호출부에서 재생성 유도)
        try:
            ts_path.unlink(missing_ok=True)
        except Exception:
            pass
        return pd.DataFrame(columns=["source", "target", "C", "S", "A"]) 
    c = compute_concurrency(ts, top_k=limit * 5)
    s = compute_sequential_strength(ts, delta_hours=1, top_k=limit * 5)
    edges = combine_scores(c, s)
    edges = edges.sort_values("A", ascending=False).head(limit)
    return edges.reset_index(drop=True)


# ---------- ML 보강: 군집(KMeans) + 마코프 전이 ----------

def build_spot_features(ts: pd.DataFrame) -> pd.DataFrame:
    # ts: [time, spot_id, value]
    ts = ts.dropna(subset=["time"]).copy()
    ts["hour"] = ts["time"].dt.hour
    ts["weekday"] = ts["time"].dt.weekday
    ts["month"] = ts["time"].dt.month

    # 시간대/요일/월별 평균 분포
    def pivot_feat(group_cols: list[str], prefix: str) -> pd.DataFrame:
        p = ts.pivot_table(index="spot_id", columns=group_cols, values="value", aggfunc="mean").fillna(0.0)
        # 멀티인덱스 열이면 플랫하게
        if isinstance(p.columns, pd.MultiIndex):
            p.columns = [f"{prefix}_{'_'.join(map(str, c))}" for c in p.columns]
        else:
            p.columns = [f"{prefix}_{c}" for c in p.columns]
        return p

    hour_p = pivot_feat(["hour"], "h")
    wday_p = pivot_feat(["weekday"], "w")
    mon_p = pivot_feat(["month"], "m")
    stat = ts.groupby("spot_id")["value"].agg(["mean", "std", "max", "sum"]).rename(columns={
        "mean": "stat_mean", "std": "stat_std", "max": "stat_max", "sum": "stat_sum"
    })
    feats = stat.join([hour_p, wday_p, mon_p], how="left").fillna(0.0)
    return feats.reset_index()


def kmeans_cluster_features(features: pd.DataFrame, k: int | None = None, random_state: int = 42) -> pd.DataFrame:
    X = features.drop(columns=["spot_id"], errors="ignore").values
    # k 자동 선택(간단): 유효 스팟 수에 따라 3~8 범위에서 sqrt 법칙 근사
    n = max(1, X.shape[0])
    if k is None:
        k = int(np.clip(round(np.sqrt(n)), 3, 8))
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    out = features[["spot_id"]].copy()
    out["cluster"] = labels.astype(int)
    return out


def compute_markov_transitions(ts: pd.DataFrame, top_per_t: int = 5) -> pd.DataFrame:
    # 시각별 상위 스팟을 전이 집합으로 보고 조건부확률 추정
    ts = ts.copy()
    ts["time"] = ts["time"].dt.floor("h")
    # t 시각의 상위
    top_t = ts.sort_values(["time", "value"], ascending=[True, False]).groupby("time").head(top_per_t)
    # t+1 시각의 상위
    lead = ts.copy()
    lead["time"] = lead["time"] + pd.to_timedelta(1, unit="h")
    top_t1 = lead.sort_values(["time", "value"], ascending=[True, False]).groupby("time").head(top_per_t)

    merged = top_t.merge(top_t1, on=["time"], how="inner", suffixes=("_i", "_j"))
    merged = merged[merged["spot_id_i"] != merged["spot_id_j"]]
    counts = merged.groupby(["spot_id_i", "spot_id_j"]).size().reset_index(name="cnt")
    denom = counts.groupby("spot_id_i")["cnt"].transform("sum").replace(0, np.nan)
    counts["S"] = (counts["cnt"] / denom).fillna(0.0)
    s_df = counts.rename(columns={"spot_id_i": "source", "spot_id_j": "target"})[["source", "target", "S"]]
    return s_df


def apply_cluster_weight(edges: pd.DataFrame, clusters: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    # 같은 군집이면 A를 (1+alpha) 배 가중, 다르면 (1-alpha)
    cmap = clusters.set_index("spot_id")["cluster"].to_dict()
    def w(row):
        ci = cmap.get(row["source"]) ; cj = cmap.get(row["target"]) 
        if ci is None or cj is None:
            return 1.0
        return (1.0 + alpha) if ci == cj else max(0.0, 1.0 - alpha)
    edges = edges.copy()
    edges["A"] = edges["A"].astype(float) * edges.apply(w, axis=1)
    return edges


def build_ml_edges(ts_path: Path | None = None, limit: int = 200) -> pd.DataFrame:
    if ts_path is None:
        ts_path = CACHE_DIR / "timeseries.parquet"
    if not ts_path.exists():
        return pd.DataFrame(columns=["source", "target", "C", "S", "A"]) 
    try:
        ts = pd.read_parquet(ts_path)
    except Exception:
        # 손상된 파케이로 추정 → 삭제하고 빈 결과 반환(상위 호출부에서 재생성 유도)
        try:
            ts_path.unlink(missing_ok=True)
        except Exception:
            pass
        return pd.DataFrame(columns=["source", "target", "C", "S", "A"]) 
    if ts.empty:
        return pd.DataFrame(columns=["source", "target", "C", "S", "A"]) 

    # 특징 + 군집
    feats = build_spot_features(ts)
    clusters = kmeans_cluster_features(feats)
    # 저장
    feats.to_parquet(CACHE_DIR / "spot_features.parquet", index=False)
    clusters.to_parquet(CACHE_DIR / "clusters.parquet", index=False)

    # 동시혼잡 + 마코프 전이
    c = compute_concurrency(ts, top_k=limit * 5)
    s = compute_markov_transitions(ts, top_per_t=5)
    edges = combine_scores(c, s, w_c=0.4, w_s=0.6)
    edges = apply_cluster_weight(edges, clusters, alpha=0.2)
    edges = edges.sort_values("A", ascending=False).head(limit).reset_index(drop=True)
    # 캐시 저장
    edges.to_parquet(CACHE_DIR / "graph_edges.parquet", index=False)
    return edges


