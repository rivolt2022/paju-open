# PAJU Story Weaver - Frontend

React + Vite로 구축된 프론트엔드 대시보드입니다.

## 설치 및 실행

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build

# 빌드 미리보기
npm run preview
```

## 환경 변수

`.env` 파일을 생성하고 다음 변수를 설정하세요:

```
VITE_API_BASE_URL=http://localhost:8000
```

## 카카오맵 API 키

카카오맵 API 키는 `index.html`에 이미 포함되어 있습니다:
- 키: 2b9ba7b168485e99557a35e5b108d873

## 주요 컴포넌트

- `Dashboard`: 메인 대시보드 컴포넌트
- `MapView`: 카카오맵 기반 관광지 위치 시각화
- `PredictionChart`: 관광지 방문 예측 차트
- `GeneratedContent`: 생성형 AI 콘텐츠 표시 및 생성

## 기술 스택

- React 18
- Vite
- react-kakao-maps-sdk
- Recharts (차트)
- Axios (HTTP 클라이언트)
