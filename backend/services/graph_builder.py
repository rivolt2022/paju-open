from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(r["source"], r["target"], C=float(r.get("C", 0.0)), S=float(r.get("S", 0.0)), A=float(r.get("A", 0.0)))
    return G


def top_edges(edges: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    if "A" not in edges.columns:
        edges["A"] = 0.0
    return edges.sort_values("A", ascending=False).head(limit).reset_index(drop=True)


