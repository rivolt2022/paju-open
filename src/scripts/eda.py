"""
파주시 데이터 탐색적 데이터 분석 (EDA) 스크립트
엑셀 파일들을 읽어 기본 통계와 패턴을 분석합니다.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "src" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_sheets_info() -> Dict[str, Any]:
    """sheets.json에서 시트 정보 로드"""
    sheets_json_path = DATA_DIR / "sheets.json"
    with open(sheets_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_excel_sheet(file_path: Path, sheet_name: str) -> pd.DataFrame:
    """엑셀 파일의 특정 시트 읽기"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"  경고: 시트 '{sheet_name}' 읽기 실패: {e}")
        return pd.DataFrame()

def analyze_dataframe(df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
    """데이터프레임 기본 통계 분석"""
    if df.empty:
        return {}
    
    info = {
        "시트명": sheet_name,
        "행수": len(df),
        "열수": len(df.columns),
        "열명": df.columns.tolist(),
        "결측치": df.isnull().sum().to_dict(),
        "결측률": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "데이터타입": df.dtypes.astype(str).to_dict(),
    }
    
    # 숫자형 컬럼 통계
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info["숫자형통계"] = df[numeric_cols].describe().to_dict()
    
    # 샘플 데이터
    info["샘플_5행"] = df.head(5).to_dict('records')
    
    return info

def analyze_file(file_path: Path, sheets_info: Dict[str, Any]) -> Dict[str, Any]:
    """단일 엑셀 파일 분석"""
    print(f"\n{'='*60}")
    print(f"분석 중: {file_path.name}")
    print(f"{'='*60}")
    
    file_analysis = {
        "파일명": file_path.name,
        "시트목록": [],
        "전체통계": {}
    }
    
    # 해당 파일의 시트 찾기
    file_sheets = [s for s in sheets_info.get("sheets", []) 
                   if s["name"].startswith(file_path.name)]
    
    if not file_sheets:
        print(f"  경고: {file_path.name}에 대한 시트 정보를 찾을 수 없습니다.")
        return file_analysis
    
    print(f"  발견된 시트 수: {len(file_sheets)}")
    
    all_rows = 0
    all_cols = set()
    
    for sheet_info in file_sheets:
        sheet_full_name = sheet_info["name"]
        # 시트명 추출 (파일명::시트명 형식)
        if "::" in sheet_full_name:
            sheet_name = sheet_full_name.split("::")[-1]
        else:
            sheet_name = sheet_full_name
        
        print(f"\n  시트: {sheet_name}")
        print(f"    - 행수: {sheet_info.get('nrows', 0)}, 열수: {sheet_info.get('ncols', 0)}")
        
        # 엑셀 파일에서 시트 읽기 시도
        df = read_excel_sheet(file_path, sheet_name)
        
        if not df.empty:
            sheet_analysis = analyze_dataframe(df, sheet_name)
            file_analysis["시트목록"].append(sheet_analysis)
            
            all_rows += len(df)
            all_cols.update(df.columns)
            
            # 숫자형 컬럼 통계 요약
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"    - 숫자형 컬럼: {len(numeric_cols)}개")
                print(f"    - 주요 통계:")
                for col in numeric_cols[:3]:  # 처음 3개만 출력
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    print(f"      {col}: 평균={mean_val:.2f}, 표준편차={std_val:.2f}")
    
    file_analysis["전체통계"] = {
        "총행수": all_rows,
        "총고유열수": len(all_cols),
        "고유열명": list(all_cols)
    }
    
    return file_analysis

def analyze_all_files():
    """모든 엑셀 파일 분석"""
    print("\n" + "="*60)
    print("파주시 데이터 탐색적 데이터 분석 (EDA) 시작")
    print("="*60)
    
    # sheets.json 로드
    sheets_info = load_sheets_info()
    
    # 엑셀 파일 목록
    excel_files = [
        "1_생활인구분석_LG유플러스.xlsx",
        "1_생활인구분석_경기데이터드림.xlsx",
        "3_관광지분석_LG유플러스.xlsx",
        "3_관광지분석_삼성카드.xlsx",
        "4_관광지트렌드분석_LG유플러스.xlsx",
        "4_관광지트렌드분석_삼성카드.xlsx",
        "5_식품위생업소소비분석_경기데이터드림.xlsx",
        "5_식품위생업소소비분석_삼성카드.xlsx",
        "6_소비경제규모추정.xlsx",
    ]
    
    all_analysis = {}
    
    for excel_file in excel_files:
        file_path = DATA_DIR / excel_file
        if file_path.exists():
            file_analysis = analyze_file(file_path, sheets_info)
            all_analysis[excel_file] = file_analysis
        else:
            print(f"\n경고: 파일을 찾을 수 없습니다: {excel_file}")
    
    # 결과 저장
    output_file = OUTPUT_DIR / "eda_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_analysis, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"분석 완료! 결과가 저장되었습니다: {output_file}")
    print(f"{'='*60}")
    
    # 요약 통계 출력
    print_summary(all_analysis)
    
    return all_analysis

def print_summary(all_analysis: Dict[str, Any]):
    """전체 분석 요약 출력"""
    print("\n" + "="*60)
    print("전체 요약 통계")
    print("="*60)
    
    total_files = len(all_analysis)
    total_sheets = sum(len(v.get("시트목록", [])) for v in all_analysis.values())
    total_rows = sum(v.get("전체통계", {}).get("총행수", 0) for v in all_analysis.values())
    
    print(f"\n총 파일 수: {total_files}개")
    print(f"총 시트 수: {total_sheets}개")
    print(f"총 데이터 행수: {total_rows:,}행")
    
    print("\n파일별 상세:")
    for file_name, analysis in all_analysis.items():
        stats = analysis.get("전체통계", {})
        print(f"\n  {file_name}:")
        print(f"    - 시트 수: {len(analysis.get('시트목록', []))}개")
        print(f"    - 총 행수: {stats.get('총행수', 0):,}행")
        print(f"    - 고유 열 수: {stats.get('총고유열수', 0)}개")

def analyze_time_series_patterns():
    """시계열 패턴 분석 (시간, 월별 추이 등)"""
    print("\n" + "="*60)
    print("시계열 패턴 분석")
    print("="*60)
    
    sheets_info = load_sheets_info()
    
    # 시간 관련 컬럼이 있는 시트 찾기
    time_keywords = ["월", "연도", "시간", "요일", "월별", "연도별"]
    
    time_series_data = []
    
    for sheet_info in sheets_info.get("sheets", []):
        sheet_name = sheet_info["name"]
        
        # 시간 관련 키워드 확인
        has_time = any(keyword in sheet_name for keyword in time_keywords)
        if has_time:
            # 파일명 추출
            if "::" in sheet_name:
                file_name = sheet_name.split("::")[0]
                sheet = sheet_name.split("::")[-1]
            else:
                continue
            
            file_path = DATA_DIR / file_name
            if file_path.exists():
                df = read_excel_sheet(file_path, sheet)
                if not df.empty:
                    # 시간 관련 컬럼 찾기
                    time_cols = [col for col in df.columns 
                               if any(keyword in str(col) for keyword in time_keywords)]
                    
                    if time_cols:
                        time_series_data.append({
                            "파일": file_name,
                            "시트": sheet,
                            "시간컬럼": time_cols,
                            "행수": len(df)
                        })
    
    print(f"\n시간 관련 데이터를 가진 시트: {len(time_series_data)}개")
    for item in time_series_data[:10]:  # 처음 10개만 출력
        print(f"  - {item['파일']} / {item['시트']}")
        print(f"    시간컬럼: {item['시간컬럼']}")

if __name__ == "__main__":
    # 전체 파일 분석
    analysis_results = analyze_all_files()
    
    # 시계열 패턴 분석
    analyze_time_series_patterns()
    
    print("\n분석이 완료되었습니다!")
