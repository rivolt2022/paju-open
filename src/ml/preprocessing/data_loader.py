"""
데이터 로더 - 엑셀 파일에서 데이터를 로드하고 통합합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

class DataLoader:
    """엑셀 데이터 로더 클래스"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Args:
            data_dir: 데이터 디렉토리 경로 (기본값: 프로젝트 루트/data)
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.sheets_info = self._load_sheets_info()
    
    def _load_sheets_info(self) -> Dict:
        """sheets.json 파일 로드"""
        sheets_json_path = self.data_dir / "sheets.json"
        if sheets_json_path.exists():
            with open(sheets_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def load_excel_file(self, filename: str) -> Dict[str, pd.DataFrame]:
        """
        엑셀 파일의 모든 시트 로드
        
        Args:
            filename: 엑셀 파일명
            
        Returns:
            시트명을 키로 하는 DataFrame 딕셔너리
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 엑셀 파일의 모든 시트 읽기
        excel_file = pd.ExcelFile(file_path)
        sheets = {}
        
        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets[sheet_name] = df
            except Exception as e:
                print(f"경고: 시트 '{sheet_name}' 읽기 실패: {e}")
        
        return sheets
    
    def load_tourist_data(self) -> pd.DataFrame:
        """관광지 분석 데이터 로드 및 통합"""
        data_frames = []
        
        # LG유플러스 관광지 분석 데이터
        lg_file = "3_관광지분석_LG유플러스.xlsx"
        lg_sheets = self.load_excel_file(lg_file)
        
        # 주요 시트 추출
        if "9p_10p_관광지별_추이" in lg_sheets:
            df = lg_sheets["9p_10p_관광지별_추이"].copy()
            df['데이터소스'] = 'LG유플러스'
            df['유형'] = '방문인구'
            data_frames.append(df)
        
        # 삼성카드 관광지 분석 데이터
        samsung_file = "3_관광지분석_삼성카드.xlsx"
        samsung_sheets = self.load_excel_file(samsung_file)
        
        if "9p_10p_관광지별_추이" in samsung_sheets:
            df = samsung_sheets["9p_10p_관광지별_추이"].copy()
            df['데이터소스'] = '삼성카드'
            df['유형'] = '매출'
            data_frames.append(df)
        
        if data_frames:
            combined = pd.concat(data_frames, ignore_index=True)
            return combined
        else:
            return pd.DataFrame()
    
    def load_population_data(self) -> pd.DataFrame:
        """생활인구 분석 데이터 로드 및 통합"""
        data_frames = []
        
        # LG유플러스 생활인구 분석
        lg_file = "1_생활인구분석_LG유플러스.xlsx"
        lg_sheets = self.load_excel_file(lg_file)
        
        if "7p 생활인구수 변화" in lg_sheets:
            df = lg_sheets["7p 생활인구수 변화"].copy()
            df['데이터소스'] = 'LG유플러스'
            data_frames.append(df)
        
        # 경기데이터드림 생활인구 분석
        gd_file = "1_생활인구분석_경기데이터드림.xlsx"
        gd_sheets = self.load_excel_file(gd_file)
        
        if "8~12p 생활인구 종류별 분석" in gd_sheets:
            df = gd_sheets["8~12p 생활인구 종류별 분석"].copy()
            df['데이터소스'] = '경기데이터드림'
            data_frames.append(df)
        
        if data_frames:
            combined = pd.concat(data_frames, ignore_index=True)
            return combined
        else:
            return pd.DataFrame()
    
    def load_consumption_data(self) -> pd.DataFrame:
        """소비 분석 데이터 로드"""
        data_frames = []
        
        # 삼성카드 식품위생업소 소비 분석
        samsung_file = "5_식품위생업소소비분석_삼성카드.xlsx"
        samsung_sheets = self.load_excel_file(samsung_file)
        
        if "5,11p 식품위생업소 변화" in samsung_sheets:
            df = samsung_sheets["5,11p 식품위생업소 변화"].copy()
            df['데이터소스'] = '삼성카드'
            data_frames.append(df)
        
        if data_frames:
            combined = pd.concat(data_frames, ignore_index=True)
            return combined
        else:
            return pd.DataFrame()
    
    def get_tourist_spots(self) -> List[str]:
        """관광지 목록 반환"""
        tourist_df = self.load_tourist_data()
        if '관광지명' in tourist_df.columns:
            spots = tourist_df['관광지명'].unique().tolist()
            return [s for s in spots if pd.notna(s)]
        return []
    
    def get_dongs(self) -> List[str]:
        """행정동 목록 반환"""
        population_df = self.load_population_data()
        if '행정동' in population_df.columns:
            dongs = population_df['행정동'].unique().tolist()
            return [d for d in dongs if pd.notna(d)]
        return []


if __name__ == "__main__":
    # 테스트
    loader = DataLoader()
    
    print("관광지 목록:")
    spots = loader.get_tourist_spots()
    print(spots[:10])
    
    print("\n행정동 목록:")
    dongs = loader.get_dongs()
    print(dongs[:10])
