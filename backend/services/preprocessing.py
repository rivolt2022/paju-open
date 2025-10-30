from __future__ import annotations

import pandas as pd
from typing import Tuple


def build_spot_table(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼이 제각각일 수 있으므로 최대한 방어적으로 표준 스키마를 만든다
    columns = {c.lower(): c for c in df.columns}
    name_candidates = [
        "name", "spot", "poi", "관광지", "관광지명", "주요 관광지", "관광지명(세부)", "시설명", "명소명", "상호명", "업소명", "업체명", "장소명",
    ]
    lat_candidates = [
        "lat", "latitude", "위도", "y", "y좌표", "ycoord", "y_coord", "ycoordinate",
    ]
    lng_candidates = [
        "lng", "lon", "longitude", "경도", "x", "x좌표", "xcoord", "x_coord", "xcoordinate",
    ]
    cat_candidates = [
        "category", "cat", "업종", "업종대분류", "업종중분류", "분류", "카테고리",
    ]

    lower_map = {c.lower(): c for c in df.columns}
    def find_col(cands):
        for key in cands:
            for col in df.columns:
                if key in col.lower():
                    return col
        return None

    name_col = find_col(name_candidates)
    lat_col = find_col(lat_candidates)
    lng_col = find_col(lng_candidates)
    cat_col = find_col(cat_candidates)

    result = pd.DataFrame()
    if name_col is None:
        # 최후의 보루: 첫 번째 컬럼을 이름으로 사용(데이터 미리보기형 시트 대응)
        if len(df.columns) > 0:
            name_col = df.columns[0]
        else:
            return result

    result["spot_id"] = df[name_col].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    result["name"] = df[name_col].astype(str)
    if lat_col is not None and lng_col is not None:
        result["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        result["lng"] = pd.to_numeric(df[lng_col], errors="coerce")
    else:
        result["lat"] = None
        result["lng"] = None
    if cat_col is not None:
        result["category"] = df[cat_col].astype(str)
    else:
        result["category"] = None
    return result.drop_duplicates(subset=["spot_id"]) 


def build_timeseries(df: pd.DataFrame, spot_key: str = "spot_id") -> pd.DataFrame:
    # 가능한 시간 컬럼 탐색
    time_candidates = [
        "time", "hour", "datetime", "date", "일시", "시간", "시간대", "시간대(시)", "기준일자", "기준년월", "연월", "기준연월", "날짜",
    ]
    # 값 후보 우선순위 그룹
    value_priority_groups = [
        ["방문인구(명)", "방문인구", "방문", "VIST_POPL_CNT_avg", "생활인구", "생활인구 전체합계", "요일별 평균 생활인구"],
        ["매출금액(원)", "이용금액", "월평균 매출금액(원)", "매출금액"],
        ["매출금액(백만원)", "월평균 매출금액(백만원)", "음식점_매출액_백만원", "음료점_매출액_백만원"],
        ["매출건수(건)", "월평균 매출건수(건)", "건수"],
    ]
    name_candidates = [
        "name", "spot", "poi", "관광지", "관광지명", "주요 관광지", "관광지명(세부)", "시설명", "명소명", "상호명", "업소명", "업체명", "장소명",
    ]

    def find_col(cands):
        for key in cands:
            for col in df.columns:
                if key in col.lower():
                    return col
        return None

    time_col = find_col(time_candidates)
    name_col = find_col(name_candidates)
    # 값 컬럼 결정
    value_col = None
    for group in value_priority_groups:
        for key in group:
            for col in df.columns:
                if key in col:
                    value_col = col
                    break
            if value_col is not None:
                break
        if value_col is not None:
            break

    if value_col is None or name_col is None:
        return pd.DataFrame(columns=["time", spot_key, "value"])  # 빈 프레임 반환

    # 시간 조립: (연도, 월[, 시간대]) → datetime
    year_col = next((c for c in df.columns if "연도" in str(c)), None)
    month_col = next((c for c in df.columns if str(c) in ["월", "연월", "기준연월", "기준년월"] or "월" in str(c)), None)
    hour_col = next((c for c in df.columns if "시간대(시)" in str(c) or str(c) in ["시간대", "시간" ]), None)

    if year_col is not None and month_col is not None:
        y = pd.to_numeric(df[year_col], errors="coerce")
        m = pd.to_numeric(df[month_col], errors="coerce")
        if hour_col is not None:
            h = pd.to_numeric(df[hour_col], errors="coerce")
        else:
            h = pd.Series(0, index=df.index)
        times = pd.to_datetime({
            "year": y.fillna(2000).astype(int),
            "month": m.fillna(1).astype(int),
            "day": 1,
            "hour": h.fillna(0).astype(int)
        }, errors="coerce")
    elif time_col is not None:
        times = pd.to_datetime(df[time_col].astype(str), errors="coerce")
    else:
        return pd.DataFrame(columns=["time", spot_key, "value"])  # 시간 정보 없음

    ts = pd.DataFrame({
        "time": times,
        spot_key: df[name_col].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna(subset=["time", "value"]).sort_values(["time", spot_key])

    # 시간 단위 정규화(시간/월 단위 그대로)
    ts["time"] = ts["time"].dt.floor("H")
    return ts


def join_spots_timeseries(spots: pd.DataFrame, ts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 키 정합성 확보: 존재하지 않는 spot은 제거
    valid_ts = ts[ts["spot_id"].isin(spots["spot_id"])].copy()
    return spots.reset_index(drop=True), valid_ts.reset_index(drop=True)


