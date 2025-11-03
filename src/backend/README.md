# PAJU Story Weaver - Backend

FastAPI 기반 백엔드 서버입니다.

## 설치

```bash
cd src/backend
pip install -r requirements.txt
```

## 실행

```bash
# 개발 서버 실행
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 주요 엔드포인트

### 데이터 API
- `GET /api/data/tourist_spots`: 관광지 목록 조회
- `GET /api/data/population/{dong}`: 행정동별 생활인구 조회

### 예측 API
- `POST /api/predict/visits`: 관광지 방문 예측
- `POST /api/predict/population`: 생활인구 패턴 예측
- `GET /api/predict/crowd_level/{spot}/{date}`: 특정 날짜 혼잡도 예측

### 생성형 AI API
- `POST /api/generate/story`: 개인화 관광 스토리 생성
- `POST /api/generate/course`: 맞춤형 코스 생성

### 분석 API
- `GET /api/analysis/trends`: 트렌드 분석 결과
- `GET /api/analysis/correlation`: 상관관계 분석 결과
