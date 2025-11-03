"""
ML 서비스 - Backend에서 ML 모델을 사용하는 서비스 레이어
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.inference import InferencePredictor, ContentGenerator
from src.ml.preprocessing import DataLoader


class MLService:
    """
    ML 모델 서비스 클래스
    Backend에서 ML 모델을 사용하는 통합 서비스 레이어
    """
    
    def __init__(self):
        """서비스 초기화"""
        # ML 모델 인스턴스 생성
        self.predictor = InferencePredictor()
        self.content_generator = ContentGenerator()
        self.data_loader = DataLoader()
    
    def predict_tourist_visits(self, tourist_spots: List[str], date: str) -> List[Dict]:
        """
        관광지 방문 예측
        
        Args:
            tourist_spots: 관광지명 리스트
            date: 날짜 (YYYY-MM-DD 형식)
            
        Returns:
            예측 결과 리스트
        """
        return self.predictor.predict_visits(tourist_spots, date)
    
    def generate_tourism_content(self, user_info: Dict, date: str, 
                                  predictions: Optional[List[Dict]] = None) -> Dict:
        """
        관광 콘텐츠 생성 (ML 예측 + 생성형 AI)
        
        Args:
            user_info: 사용자 정보
            date: 날짜
            predictions: 예측 결과 (None이면 자동 예측)
            
        Returns:
            생성된 콘텐츠 딕셔너리
        """
        # 예측 결과가 없으면 먼저 예측 수행
        if predictions is None:
            spots = self.data_loader.get_tourist_spots()
            if not spots:
                spots = ["헤이리예술마을", "DMZ평화관광", "마장호수출렁다리"]
            
            predictions = self.predict_tourist_visits(spots[:3], date)
        
        # 생성형 AI로 콘텐츠 생성
        content = self.content_generator.generate_story(user_info, predictions, date)
        
        return content
    
    def get_tourist_spots_list(self) -> List[str]:
        """관광지 목록 조회"""
        spots = self.data_loader.get_tourist_spots()
        if not spots:
            return ["헤이리예술마을", "DMZ평화관광", "마장호수출렁다리", "파주출판단지"]
        return spots
    
    def predict_population(self, dong: str, date: str) -> Dict:
        """생활인구 예측"""
        return self.predictor.predict_population(dong, date)


# 전역 서비스 인스턴스 (싱글톤 패턴)
_ml_service_instance = None

def get_ml_service() -> MLService:
    """ML 서비스 인스턴스 반환 (싱글톤)"""
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLService()
    return _ml_service_instance
