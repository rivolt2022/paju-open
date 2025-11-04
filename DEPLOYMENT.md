# 배포 가이드

이 문서는 PAJU Culture Lab 프로젝트를 Vercel(프론트엔드)과 Fly.io(백엔드)에 배포하는 방법을 설명합니다.

## 📋 목차

1. [프론트엔드 배포 (Vercel)](#프론트엔드-배포-vercel)
2. [백엔드 배포 (Fly.io)](#백엔드-배포-flyio)
3. [환경 변수 설정](#환경-변수-설정)
4. [배포 후 확인](#배포-후-확인)

---

## 프론트엔드 배포 (Vercel)

### 1. Vercel 계정 준비

1. [Vercel](https://vercel.com)에 가입/로그인
2. GitHub/GitLab/Bitbucket 계정 연결

### 2. 프로젝트 배포

#### 방법 1: Vercel CLI 사용

```bash
# Vercel CLI 설치
npm i -g vercel

# 프로젝트 루트에서 실행
cd src/frontend

# 배포
vercel

# 프로덕션 배포
vercel --prod
```

#### 방법 2: Vercel 웹 대시보드 사용

1. [Vercel Dashboard](https://vercel.com/dashboard) 접속
2. "Add New Project" 클릭
3. GitHub 저장소 선택
4. **Root Directory 설정**: `src/frontend`로 설정
5. Framework Preset: Vite 선택
6. Build Command: `npm run build` (자동 감지)
7. Output Directory: `dist` (자동 감지)
8. Environment Variables 설정 (아래 참고)
9. "Deploy" 클릭

### 3. 환경 변수 설정

Vercel 대시보드에서 환경 변수 추가:

- `VITE_API_BASE_URL`: 백엔드 API URL (예: `https://YOUR_APP_NAME.fly.dev`)

**설정 위치**: Vercel 프로젝트 설정 → Environment Variables

### 4. vercel.json 설정 확인

`src/frontend/vercel.json` 파일에서 다음 항목을 수정:

```json
{
  "rewrites": [
    {
      "source": "/api/(.*)",
      "destination": "https://YOUR_APP_NAME.fly.dev/api/$1"
    }
  ]
}
```

`YOUR_APP_NAME`을 실제 Fly.io 앱 이름으로 변경하세요.

---

## 백엔드 배포 (Fly.io)

### 1. Fly.io 계정 준비

1. [Fly.io](https://fly.io)에 가입/로그인
2. [Fly.io CLI 설치](https://fly.io/docs/hands-on/install-flyctl/):

```bash
# Windows (PowerShell)
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# macOS/Linux
curl -L https://fly.io/install.sh | sh
```

3. Fly.io 로그인:

```bash
fly auth login
```

### 2. 프로젝트 초기화

```bash
# 프로젝트 루트에서 실행
cd /path/to/paju-open

# Fly.io 앱 생성 (이미 생성된 경우 건너뛰기)
fly launch

# 또는 기존 앱 사용
fly apps create paju-culture-lab-api
```

### 3. fly.toml 설정 확인

프로젝트 루트의 `fly.toml` 파일을 확인하고 수정:

```toml
app = "paju-culture-lab-api"  # 앱 이름 변경 가능
primary_region = "icn"  # 서울 리전
```

### 4. 환경 변수 설정

```bash
# 환경 변수 설정
fly secrets set UPSTAGE_API_KEY=your_api_key_here

# 여러 환경 변수 한 번에 설정
fly secrets set \
  UPSTAGE_API_KEY=your_api_key \
  PORT=8000 \
  PYTHONUNBUFFERED=1
```

### 5. Dockerfile 빌드 및 배포

```bash
# Docker 이미지 빌드 및 배포
fly deploy

# 또는 빌드만 수행
fly deploy --build-only

# 로그 확인
fly logs

# 앱 상태 확인
fly status
```

### 6. 데이터 파일 업로드 (선택사항)

ML 모델이나 데이터 파일이 필요한 경우:

```bash
# 볼륨 생성 (필요한 경우)
fly volumes create data --size 1 --region icn

# 데이터 파일 업로드
fly ssh console
# SSH 콘솔에서 파일 업로드
```

또는 Dockerfile에서 COPY 명령으로 포함 가능합니다.

---

## 환경 변수 설정

### 프론트엔드 (Vercel)

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `VITE_API_BASE_URL` | 백엔드 API URL | `https://paju-culture-lab-api.fly.dev` |

**설정 방법**: Vercel 대시보드 → 프로젝트 설정 → Environment Variables

### 백엔드 (Fly.io)

| 변수명 | 설명 | 필수 여부 |
|--------|------|----------|
| `UPSTAGE_API_KEY` | 업스테이지 Solar Pro2 API 키 | 권장 |
| `PORT` | 서버 포트 (기본값: 8000) | 선택 |
| `PYTHONUNBUFFERED` | Python 출력 버퍼링 비활성화 | 선택 |

**설정 방법**:

```bash
# 개별 설정
fly secrets set UPSTAGE_API_KEY=your_api_key

# 확인
fly secrets list

# 삭제
fly secrets unset UPSTAGE_API_KEY
```

---

## 배포 후 확인

### 1. 백엔드 Health Check

```bash
# Health check 엔드포인트 확인
curl https://YOUR_APP_NAME.fly.dev/health

# 예상 응답
{
  "status": "healthy",
  "service": "PAJU Culture Lab API",
  "version": "1.0.0"
}
```

### 2. 백엔드 API 문서 확인

브라우저에서 접속:
- Swagger UI: `https://YOUR_APP_NAME.fly.dev/docs`
- ReDoc: `https://YOUR_APP_NAME.fly.dev/redoc`

### 3. 프론트엔드 연결 확인

1. Vercel에서 배포된 URL 접속
2. 브라우저 개발자 도구(F12) → Network 탭
3. API 호출이 정상적으로 이루어지는지 확인

### 4. 로그 확인

#### Vercel 로그

Vercel 대시보드 → 프로젝트 → Deployments → Functions Logs

#### Fly.io 로그

```bash
# 실시간 로그 확인
fly logs

# 특정 시간대 로그
fly logs --region icn

# 앱 상태 확인
fly status

# SSH 접속 (디버깅)
fly ssh console
```

---

## 트러블슈팅

### 백엔드 배포 문제

1. **포트 오류**: `fly.toml`에서 `internal_port` 확인
2. **의존성 설치 실패**: `requirements.txt` 확인, Dockerfile의 pip 명령 확인
3. **모델 파일 없음**: `src/ml/models/saved/spatiotemporal_model.pkl` 파일 확인

### 프론트엔드 배포 문제

1. **빌드 실패**: `package.json`의 빌드 스크립트 확인
2. **API 연결 오류**: `VITE_API_BASE_URL` 환경 변수 확인
3. **라우팅 문제**: `vercel.json`의 rewrites 설정 확인

### CORS 오류

백엔드 `main.py`에서 CORS 설정 확인:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-vercel-app.vercel.app"],  # Vercel URL 추가
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 참고 사항

### Fly.io 리전

현재 설정: `icn` (서울, 인천)
다른 리전 사용 시 `fly.toml`의 `primary_region` 변경

### Fly.io 무료 플랜 제한

- 공유 CPU (256MB 메모리)
- 3개 앱까지
- 월 160GB 시간 제한

### Vercel 무료 플랜 제한

- 100GB 대역폭
- 무제한 배포
- 자동 HTTPS

---

## 배포 체크리스트

- [ ] Vercel 계정 생성 및 연결
- [ ] Fly.io 계정 생성 및 CLI 설치
- [ ] `vercel.json`에서 Fly.io 앱 이름 설정
- [ ] `fly.toml`에서 앱 이름 및 리전 확인
- [ ] 환경 변수 설정 (프론트엔드, 백엔드)
- [ ] Dockerfile 빌드 테스트
- [ ] Health check 엔드포인트 확인
- [ ] API 문서 접근 확인
- [ ] 프론트엔드-백엔드 연결 테스트
- [ ] CORS 설정 확인

---

## 추가 리소스

- [Vercel 문서](https://vercel.com/docs)
- [Fly.io 문서](https://fly.io/docs)
- [FastAPI 배포 가이드](https://fastapi.tiangolo.com/deployment/)
- [Vite 배포 가이드](https://vitejs.dev/guide/static-deploy.html)

