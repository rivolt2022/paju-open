"""주말 패턴 데이터 확인 스크립트"""
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.preprocessing.enhanced_data_loader import EnhancedDataLoader

loader = EnhancedDataLoader()

# 각 시트를 개별적으로 확인
print("="*60)
print("주말 패턴 데이터 - 요일별 생활인구 현황")
print("="*60)
weekend_df1 = loader.load_excel_sheet("1_생활인구분석_LG유플러스.xlsx", "29p 요일별 생활인구 현황")
print(f"데이터 행 수: {len(weekend_df1)}")
if not weekend_df1.empty:
    print(f"컬럼: {weekend_df1.columns.tolist()}")
    print(weekend_df1.head(10))

print("\n" + "="*60)
print("주말 패턴 데이터 - 요일별 관광지 방문 패턴")
print("="*60)
weekend_df2 = loader.load_excel_sheet("3_관광지분석_LG유플러스.xlsx", "15p_27p_39p_51p_63p_요일별")
print(f"데이터 행 수: {len(weekend_df2)}")
if not weekend_df2.empty:
    print(f"컬럼: {weekend_df2.columns.tolist()}")
    print(weekend_df2.head(10))

# 전체 데이터 로드
weekend_df = loader.load_weekend_pattern_data()

print("="*60)
print("주말 패턴 데이터 분석")
print("="*60)
print(f"\n데이터 행 수: {len(weekend_df)}")
print(f"\n컬럼 목록:")
print(weekend_df.columns.tolist())

print(f"\n데이터 샘플 (처음 10행):")
print(weekend_df.head(10))

print(f"\n데이터 타입:")
print(weekend_df.dtypes)

# 요일 컬럼 찾기
day_col = None
for col in weekend_df.columns:
    if '요일' in str(col) or 'day' in str(col).lower():
        day_col = col
        print(f"\n요일 컬럼 발견: {col}")
        print(f"요일 값들: {weekend_df[col].unique()}")
        break

# 방문인구/생활인구 컬럼 찾기
visit_cols = []
for col in weekend_df.columns:
    if '방문인구' in str(col) or '생활인구' in str(col) or '인구' in str(col):
        if pd.api.types.is_numeric_dtype(weekend_df[col]):
            visit_cols.append(col)
            print(f"\n숫자형 인구 컬럼 발견: {col}")
            print(f"평균값: {weekend_df[col].mean():.0f}")
            print(f"최소값: {weekend_df[col].min():.0f}")
            print(f"최대값: {weekend_df[col].max():.0f}")

if day_col and visit_cols:
    print(f"\n주말/평일 분석:")
    # 요일별로 그룹화
    print(f"\n요일별 데이터:")
    for day in weekend_df[day_col].unique():
        if pd.notna(day):
            day_data = weekend_df[weekend_df[day_col] == day]
            print(f"\n  [{day}]")
            for visit_col in visit_cols:
                values = day_data[visit_col].dropna()
                if len(values) > 0:
                    print(f"    {visit_col}: 평균 {values.mean():.0f}, 최소 {values.min():.0f}, 최대 {values.max():.0f}")
    
    # 주말/평일 구분
    weekend_days = ['토', '일', '토요일', '일요일', 'Saturday', 'Sunday']
    weekend_mask = weekend_df[day_col].astype(str).str.contains('|'.join(weekend_days), case=False, na=False)
    
    print(f"\n주말/평일 통계:")
    for visit_col in visit_cols:
        weekend_values = weekend_df[weekend_mask][visit_col].dropna()
        weekday_values = weekend_df[~weekend_mask][visit_col].dropna()
        
        if len(weekend_values) > 0 and len(weekday_values) > 0:
            weekend_avg = weekend_values.mean()
            weekday_avg = weekday_values.mean()
            ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 0
            
            print(f"\n[{visit_col}]")
            print(f"  평일 평균: {weekday_avg:.0f} (n={len(weekday_values)})")
            print(f"  주말 평균: {weekend_avg:.0f} (n={len(weekend_values)})")
            print(f"  주말/평일 비율: {ratio:.2f} (주말이 평일보다 {ratio:.1f}배)")
        else:
            print(f"\n[{visit_col}]")
            print(f"  주말 데이터: {len(weekend_values)}개")
            print(f"  평일 데이터: {len(weekday_values)}개")

