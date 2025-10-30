### 세션 히스토리 (2025-10-30)

- **프로젝트**: paju-open
- **목표**: `task/idea.md` 계획을 실제 구현 (백엔드 ETL/지표/그래프/코스, FastAPI) + 프런트(React, Kakao Map) + ML 기반 자동 최적 코스(클러스터링 + 마코프)
- **현 상태 요약**: 백엔드/프런트 뼈대 구축 완료. 데이터 ETL과 시계열·그래프·코스 계산 파이프라인 구현. Kakao Map으로 전환. ML 보강(클러스터링, 마코프)을 반영. 주요 런타임 오류들 수정. 최소 2노드 코스 폴백 및 손상된 Parquet 자동 치유 로직 추가. 현재 지도 표시 OK, 데이터 노출은 캐시 재생성 후 정상화 기대.

### 핵심 변경 내역
- **백엔드 구조**: `backend/` 패키지화(`__init__.py`), FastAPI 엔드포인트 구현
  - `backend/app.py`
    - 엔드포인트: `/api/health`, `/api/spots`, `/api/graph`, `/api/courses`, `/api/debug/sheets`
    - 캐시 비어있을 때 `build_caches_from_excels()` 트리거
    - `/api/courses`: 경로 탐색 결과가 없을 때 `fallback_two_node_paths`로 2노드 코스 폴백 추가
  - `backend/models/schemas.py`: Pydantic 스키마 정의
  - `backend/services/data_loader.py`: 모든 시트 로드(openpyxl), 캐시(parquet) 빌드/로드
  - `backend/services/preprocessing.py`: 컬럼 후보 확장(한글), 이름 컬럼 폴백, 시계열 생성의 타입 안정성 강화
  - `backend/services/metrics.py`:
    - C/S/A 계산 파이프라인
    - KMeans 클러스터링, 마코프 전이 확률 추정
    - 상관행렬 인덱스 충돌 수정(`source`/`target` 명시)
    - 시간 단위 경고 해결(`H`→`h`), 손상 Parquet 자동 삭제 및 안전 복구
  - `backend/services/graph_builder.py`, `backend/services/course_recommender.py`: 그래프/코스 유틸과 폴백 경로 제공

- **프런트엔드**: Vite + React + TS, Kakao Map으로 전환
  - `frontend/package.json`: `@vitejs/plugin-react`, `react-kakao-maps-sdk` 추가
  - `frontend/index.html`: Kakao Map SDK 스크립트(API 키 적용)
  - `frontend/src/components/MapView.tsx`: Kakao Map(`Map`, `MapMarker`, `Polyline`)로 스팟/엣지 표시
  - `frontend/src/components/CourseCard.tsx`: 추천 코스 카드 UI
  - `frontend/src/lib/api.ts`: 백엔드 API 연동

### 주요 오류와 조치
- ImportError: 패키지 인식 실패 → `__init__.py` 추가, `uvicorn backend.app:app`로 실행
- Vite 플러그인 미설치 → `@vitejs/plugin-react` 설치 안내
- Mapbox v7 API 변경 → `StaticMap`→`Map`
- Mapbox 토큰 누락 → `.env.local` 안내 (이후 Kakao로 전환)
- `/api/spots` 빈 응답 문제
  - 모든 시트 로드, 한글 컬럼 후보 확장, 이름 컬럼 폴백, 좌표 보정 로직 보강, 빈 캐시 시 자동 재빌드
- `AttributeError: 'int' object has no attribute 'fillna'`
  - 시계열 시간 컬럼 처리에서 `h`를 항상 Series로 보장
- 상관행렬 reset_index 중복 컬럼 → 레벨명 `source`/`target` 명시해 충돌 제거
- 추천 경로 없음 → 2노드 코스 폴백 추가
- Parquet 손상(4 bytes) → 읽기 예외 시 자동 삭제 후 재생성 유도
- pandas FutureWarning → 시간 단위 `'H'`를 `'h'`로 교체
- KMeans Windows MKL 경고 → `OMP_NUM_THREADS=1` 안내

### 데이터 인식(주요 컬럼 매핑)
- 스팟명 후보: `관광지명`, `관광지명(세부)`, `관광지명.1`, `주요 관광지`, 필요 시 첫 컬럼 폴백
- 시간 파생: `연도`, `월`, `시간대(시)` 기준의 시계열 생성
- 값 컬럼: 카드/통신 지표에서 `매출금액(원|백만원)`, `매출건수(건)`, `방문인구(명)` 등 스케일 표준화

### 실행/재기동 가이드(Windows PowerShell)
```powershell
Ctrl+C
$env:OMP_NUM_THREADS = "1"
$env:PYTHONPATH = "$PWD\backend"
uvicorn backend.app:app --reload --port 8000
```

### 점검 체크리스트
- 백엔드 API
  - `/api/debug/sheets`: 로드된 시트/컬럼 확인
  - `/api/spots`: 스팟 배열 존재 여부
  - `/api/graph`: 엣지 상위 N 확인
  - `/api/courses`: 3~5노드 경로 또는 2노드 폴백 존재 여부
- 프런트
  - Kakao Map 표시, 마커/폴리라인 가시성, 코스 카드 렌더링

### 잔여/권장 작업
- 상위 엣지 100개 시각화 검증 및 캡처
- 좌표 지오코딩/보정(필요 시) 자동화
- 프런트 스타일 다듬기(테마 토글, 로딩/에러 상태)
- 캐시 재생성 버튼/플래그(운영 편의)

### 변경된 파일(중요)
- `backend/app.py` (코스 폴백 추가)
- `backend/services/metrics.py` (상관행렬 충돌 수정, 시간 단위 변경, Parquet 자가치유)
- 기타: 로더/전처리/프런트 Kakao Map 전환 관련 파일 일괄 수정

### 참고 명령
- 프런트
  - `cd frontend && pnpm install && pnpm run dev`
- 파이썬 패키지
  - `pip install fastapi uvicorn[standard] pandas numpy scikit-learn openpyxl pyarrow`

### 비고
- 데이터 컬럼명이 문서와 상이할 수 있어 후보군/폴백을 다수 적용했으며, 필요 시 `preprocessing.py`의 매핑 후보를 추가 확장 가능.
