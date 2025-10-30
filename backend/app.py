from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .models.schemas import SpotResponse, GraphEdge, CourseResponse
from .core.config import CACHE_DIR
from .services.data_loader import build_caches_from_excels
from .services.metrics import build_edges_from_timeseries, build_ml_edges
from .services.graph_builder import top_edges
from .services.course_recommender import simple_paths_from_edges, fallback_two_node_paths
from .services.data_loader import load_excel_sources
import pandas as pd
import numpy as np

app = FastAPI(title="Paju Open - Course Recommender", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/spots", response_model=SpotResponse)
def get_spots(time: str = Query(None, description="ISO-like time e.g. 2024-01-01T10")):
    cache_path = CACHE_DIR / "spots.parquet"
    if not cache_path.exists():
        build_caches_from_excels()
    if cache_path.exists():
        spots_df = pd.read_parquet(cache_path)
        if spots_df.empty:
            # 캐시가 비어있으면 재생성 시도
            build_caches_from_excels()
            if cache_path.exists():
                spots_df = pd.read_parquet(cache_path)
        # 좌표가 거의 없으면 임시 좌표 부여(데모용)
        if spots_df[["lat", "lng"]].isna().mean().mean() > 0.8 and len(spots_df) > 0:
            center_lat, center_lng = 37.75, 126.78
            def jitter(s: str):
                seed = abs(hash(s)) % 10_000
                rng = np.random.default_rng(seed)
                return center_lat + (rng.random() - 0.5) * 0.2, center_lng + (rng.random() - 0.5) * 0.2
            lats, lngs = [], []
            for sid in spots_df["spot_id"].astype(str).tolist():
                la, ln = jitter(sid)
                lats.append(la)
                lngs.append(ln)
            spots_df["lat"] = lats
            spots_df["lng"] = lngs
        spots = [
            {
                "spot_id": r["spot_id"],
                "name": r["name"],
                "lat": None if pd.isna(r.get("lat")) else float(r.get("lat")),
                "lng": None if pd.isna(r.get("lng")) else float(r.get("lng")),
                "category": None if pd.isna(r.get("category")) else str(r.get("category")),
                "metrics": {},
            }
            for _, r in spots_df.iterrows()
        ]
        return SpotResponse(spots=spots, meta={"time": time})
    return SpotResponse(spots=[], meta={"time": time})


@app.get("/api/debug/sheets")
def debug_sheets():
    """엑셀에서 감지된 시트와 컬럼 요약을 반환"""
    sources = load_excel_sources()
    summary = []
    for name, df in sources.items():
        cols = list(map(str, df.columns.tolist()))
        head = df.head(3).astype(str).to_dict(orient="records")
        summary.append({"name": name, "nrows": len(df), "ncols": len(cols), "columns": cols, "head": head})
    return {"sheets": summary}


@app.get("/api/graph", response_model=list[GraphEdge])
def get_graph(time: str = Query(None, description="ISO-like time e.g. 2024-01-01T10"), limit: int = 100):
    # ML 결과가 있으면 우선 사용
    ml_edges_path = CACHE_DIR / "graph_edges.parquet"
    if not ml_edges_path.exists():
        # 캐시/엣지 재구축
        build_caches_from_excels()
    if ml_edges_path.exists():
        edges = pd.read_parquet(ml_edges_path)
        edges = edges.sort_values("A", ascending=False).head(limit)
    else:
        cache_path = CACHE_DIR / "timeseries.parquet"
        if not cache_path.exists():
            build_caches_from_excels()
        edges = build_edges_from_timeseries(cache_path, limit=limit)
    edges = top_edges(edges, limit=limit)
    return [GraphEdge(**r._asdict()) if hasattr(r, "_asdict") else GraphEdge(**r) for r in edges.to_dict(orient="records")]


@app.get("/api/courses", response_model=CourseResponse)
def get_courses(theme: str = Query("family"), time: str = Query(None), limit: int = 100):
    ml_edges_path = CACHE_DIR / "graph_edges.parquet"
    if not ml_edges_path.exists():
        build_caches_from_excels()
    if ml_edges_path.exists():
        edges = pd.read_parquet(ml_edges_path).sort_values("A", ascending=False).head(limit)
    else:
        cache_path = CACHE_DIR / "timeseries.parquet"
        if not cache_path.exists():
            build_caches_from_excels()
        edges = build_edges_from_timeseries(cache_path, limit=limit)
    paths = simple_paths_from_edges(edges, max_len=5, min_len=3, k=3)
    if len(paths) == 0:
        # 연결성이 약할 때 2노드 코스로 폴백
        paths = fallback_two_node_paths(edges, k=3)
    courses = [
        {
            "path": [{"spot_id": sid, "order": i + 1} for i, sid in enumerate(p)],
            "value": float(edges[edges.apply(lambda x: {x["source"], x["target"]}.issubset(set(p)), axis=1)]["A"].sum()) if not edges.empty else None,
        }
        for p in paths
    ]
    return CourseResponse(theme=theme, time=time, courses=courses)


