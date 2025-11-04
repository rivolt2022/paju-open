import { useState, useEffect } from 'react'
import { MdPeople, MdGpsFixed, MdBusiness, MdAccessTime, MdLightbulb, MdCalendarToday, MdMenuBook, MdDateRange, MdPlayArrow } from 'react-icons/md'
import './HeroSection.css'

export default function HeroSection({ statistics, predictions, trendData, onDateChange, onTimeSlotChange, selectedDate, selectedTimeSlot, onPeriodPredict }) {
  const [animatedValue, setAnimatedValue] = useState(0)
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    // 시간 업데이트
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    // 애니메이션 값 업데이트
    const animationInterval = setInterval(() => {
      setAnimatedValue(prev => (prev + 1) % 360)
    }, 50)

    return () => {
      clearInterval(timeInterval)
      clearInterval(animationInterval)
    }
  }, [])

  const formatTime = (date) => {
    return date.toLocaleTimeString('ko-KR', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const formatDate = (date) => {
    return date.toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      weekday: 'long'
    })
  }

  // statistics에서 모델 정확도 가져오기 (ML 모델 분석 결과만 사용)
  const modelAccuracy = statistics?.model_accuracy 
    ? (statistics.model_accuracy * 100).toFixed(1) 
    : null
  
  // 정확도 레벨 결정 (값이 있을 때만)
  const accuracyLevel = modelAccuracy 
    ? (parseFloat(modelAccuracy) >= 99 ? 'excellent' : parseFloat(modelAccuracy) >= 95 ? 'high' : 'good')
    : null

  const totalVisits = statistics?.total_visits?.toLocaleString() || null
  const activeSpaces = statistics?.active_spaces || null

  // 전일 대비 계산 (trendData 사용)
  const dailyTrend = trendData?.daily_trend || []
  const yesterdayChange = dailyTrend.length >= 2 
    ? ((dailyTrend[dailyTrend.length - 1].visits - dailyTrend[dailyTrend.length - 2].visits) / dailyTrend[dailyTrend.length - 2].visits * 100).toFixed(1)
    : null

  // 최적 방문 시간대 (predictions에서 가져오기)
  const optimalTime = predictions?.predictions && predictions.predictions.length > 0
    ? predictions.predictions[0].optimal_time || null
    : null

  // 기간 선택 상태
  const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0])
  const [endDate, setEndDate] = useState(() => {
    const tomorrow = new Date()
    tomorrow.setDate(tomorrow.getDate() + 7)
    return tomorrow.toISOString().split('T')[0]
  })
  const [predicting, setPredicting] = useState(false)
  const [hasInitialPrediction, setHasInitialPrediction] = useState(false)

  const handlePeriodPredict = async () => {
    if (!startDate || !endDate || startDate > endDate) {
      alert('올바른 기간을 선택해주세요.')
      return
    }
    setPredicting(true)
    if (onPeriodPredict) {
      await onPeriodPredict(startDate, endDate)
    }
    setPredicting(false)
  }

  // 페이지 로드 시 기본 예측 수행 (한 번만)
  useEffect(() => {
    if (hasInitialPrediction) return // 이미 실행했으면 더 이상 실행하지 않음
    
    if (startDate && endDate && onPeriodPredict) {
      setHasInitialPrediction(true)
      const performInitialPrediction = async () => {
        if (!startDate || !endDate || startDate > endDate) {
          return
        }
        setPredicting(true)
        try {
          if (onPeriodPredict) {
            await onPeriodPredict(startDate, endDate)
          }
        } catch (error) {
          console.error('초기 예측 실패:', error)
        } finally {
          setPredicting(false)
        }
      }
      // 약간의 지연을 두어 무한루프 방지
      const timer = setTimeout(() => {
        performInitialPrediction()
      }, 500)
      return () => clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // 컴포넌트 마운트 시 한 번만 실행

  return (
    <div className="hero-section">
      <div className="hero-background">
        <div className="hero-gradient"></div>
        <div className="hero-particles">
          {[...Array(20)].map((_, i) => (
            <div 
              key={i} 
              className="particle"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 3}s`,
                animationDuration: `${3 + Math.random() * 4}s`
              }}
            />
          ))}
        </div>
      </div>

      <div className="hero-content">
        <div className="hero-main">
          {/* 타이틀 섹션 */}
          <div className="hero-title-section">
            <div className="hero-badge">
              <MdMenuBook className="badge-icon" />
              <span>PAJU Culture Lab</span>
            </div>
            <h1 className="hero-title">
              <span className="title-line">데이터 기반</span>
              <span className="title-line highlight">문화 콘텐츠 큐레이터 AI</span>
            </h1>
            <p className="hero-description">
              AI 문화 및 콘텐츠 서비스를 통한 출판단지 활성화<br />
              생성형 AI와 ML 예측으로 문화 공간의 미래를 설계합니다
            </p>
          </div>

          {/* 예측 기간 섹션 */}
          <div className="hero-period-section">
            <div className="period-card">
              <div className="period-header">
                <MdDateRange className="period-header-icon" />
                <span className="period-header-title">예측 기간</span>
              </div>
              <div className="period-content">
                <div className="date-inputs">
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="date-input"
                  />
                  <span className="date-separator">~</span>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    min={startDate}
                    className="date-input"
                  />
                </div>
                <button
                  className="predict-button"
                  onClick={handlePeriodPredict}
                  disabled={predicting}
                >
                  {predicting ? (
                    <>
                      <span className="spinner"></span>
                      <span>예측 중...</span>
                    </>
                  ) : (
                    <>
                      <MdPlayArrow className="predict-icon" />
                      <span>예측 실행</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* 통계 카드 섹션 */}
          <div className="hero-stats-section">
            <div className="hero-stats">
              <div className="hero-stat-card">
                <div className="stat-icon-wrapper">
                  <div className="stat-icon"><MdPeople /></div>
                </div>
                <div className="stat-content">
                  <div className="stat-value">{totalVisits !== null ? totalVisits : '-'}</div>
                  <div className="stat-label">오늘 예상 방문자</div>
                </div>
              </div>

              <div className="hero-stat-card">
                <div className="stat-icon-wrapper">
                  <div className="stat-icon"><MdGpsFixed /></div>
                </div>
                <div className="stat-content">
                  <div className={`stat-value ${modelAccuracy ? `stat-accuracy-${accuracyLevel}` : ''}`}>
                    {modelAccuracy !== null ? `${modelAccuracy}%` : '-'}
                  </div>
                  <div className="stat-label">예측 신뢰도</div>
                  {modelAccuracy && (
                    <div className="stat-hint">
                      {accuracyLevel === 'excellent' 
                        ? '날씨 예보 수준으로 매우 신뢰할 수 있습니다' 
                        : accuracyLevel === 'high'
                        ? '날씨 예보처럼 신뢰할 수 있는 수준입니다'
                        : '신뢰할 수 있는 수준입니다'}
                    </div>
                  )}
                </div>
              </div>

              <div className="hero-stat-card">
                <div className="stat-icon-wrapper">
                  <div className="stat-icon"><MdBusiness /></div>
                </div>
                <div className="stat-content">
                  <div className="stat-value">{activeSpaces !== null ? activeSpaces : '-'}</div>
                  <div className="stat-label">활성 문화 공간</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="hero-sidebar">
          {/* 시간 카드 */}
          <div className="hero-time-card">
            <div className="time-icon"><MdAccessTime /></div>
            <div className="time-content">
              <div className="time-display">{formatTime(currentTime)}</div>
              <div className="date-display">{formatDate(currentTime)}</div>
            </div>
          </div>

          {/* 실시간 인사이트 카드 */}
          <div className="hero-quick-insights">
            <div className="insight-header">
              <span className="insight-icon"><MdLightbulb /></span>
              <span>실시간 인사이트</span>
            </div>
            <div className="insight-list">
              {yesterdayChange !== null && (
                <div className="insight-item">
                  <span className="insight-dot"></span>
                  <span>오늘 방문 예측: 전일 대비 <strong>{parseFloat(yesterdayChange) > 0 ? '+' : ''}{yesterdayChange}%</strong></span>
                </div>
              )}
              {optimalTime && (
                <div className="insight-item">
                  <span className="insight-dot"></span>
                  <span>최적 방문 시간대: <strong>{optimalTime}</strong></span>
                </div>
              )}
              {modelAccuracy !== null && (
                <div className="insight-item">
                  <span className="insight-dot"></span>
                  <span>예측 신뢰도: <strong>{modelAccuracy}%</strong> ({accuracyLevel === 'excellent' ? '매우 높음 (날씨 예보 수준)' : accuracyLevel === 'high' ? '높음' : '양호'})</span>
                </div>
              )}
              {!yesterdayChange && !optimalTime && modelAccuracy === null && (
                <div className="insight-item">
                  <span className="insight-dot"></span>
                  <span>데이터 로딩 중...</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

    </div>
  )
}

