from __future__ import annotations

import pandas as pd
from typing import List


def simple_paths_from_edges(edges: pd.DataFrame, max_len: int = 5, min_len: int = 3, k: int = 3) -> List[list[str]]:
    # 매우 단순한 휴리스틱: 상위 엣지들을 이어 붙여 경로 생성(사이클 방지)
    edges = edges.sort_values("A", ascending=False).copy()
    paths: List[list[str]] = []
    used_nodes: set[str] = set()

    for _, r in edges.iterrows():
        a, b = r["source"], r["target"]
        if a in used_nodes or b in used_nodes:
            continue
        path = [a, b]
        # 확장 시도: 주변 엣지로 앞/뒤로 한 노드씩
        for _ in range(max_len - 2):
            extended = False
            for _, rr in edges.iterrows():
                x, y = rr["source"], rr["target"]
                if x in path or y in path:
                    continue
                # 양끝에 연결 가능한 경우 붙이기
                if path[0] in (x, y):
                    other = y if path[0] == x else x
                    path.insert(0, other)
                    extended = True
                    break
                if path[-1] in (x, y):
                    other = y if path[-1] == x else x
                    path.append(other)
                    extended = True
                    break
            if not extended:
                break
        if len(path) >= min_len:
            paths.append(path)
            used_nodes.update(path)
        if len(paths) >= k:
            break
    return paths


def fallback_two_node_paths(edges: pd.DataFrame, k: int = 3) -> List[list[str]]:
    paths: List[list[str]] = []
    used: set[str] = set()
    for _, r in edges.sort_values("A", ascending=False).iterrows():
        a, b = r["source"], r["target"]
        if a in used or b in used:
            continue
        paths.append([a, b])
        used.update([a, b])
        if len(paths) >= k:
            break
    return paths


