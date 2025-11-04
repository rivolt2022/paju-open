import { useState, useEffect } from 'react'
import { MdPeople, MdGpsFixed, MdBusiness, MdAccessTime, MdLightbulb, MdCalendarToday, MdMenuBook, MdDateRange, MdPlayArrow } from 'react-icons/md'
import './HeroSection.css'

export default function HeroSection({ statistics, predictions, trendData, onPeriodPredict, startDate: propStartDate, endDate: propEndDate, onDateChange, selectedDate = null }) {
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
  
  // 날짜 레이블 생성
  const dateLabel = selectedDate ? new Date(selectedDate).toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' }) : '오늘'

  // 날짜는 props로 받아서 사용 (Dashboard에서 관리)
  // endDate 기본값 계산
  const getDefaultEndDate = () => {
    const tomorrow = new Date()
    tomorrow.setDate(tomorrow.getDate() + 7)
    return tomorrow.toISOString().split('T')[0]
  }
  
  const [startDate, setStartDate] = useState(propStartDate || new Date().toISOString().split('T')[0])
  const [endDate, setEndDate] = useState(propEndDate || getDefaultEndDate())
  const [predicting, setPredicting] = useState(false)
  
  // props가 변경되면 로컬 상태 업데이트
  useEffect(() => {
    if (propStartDate) setStartDate(propStartDate)
    if (propEndDate) setEndDate(propEndDate)
  }, [propStartDate, propEndDate])

  const handleDateInputChange = (type, value) => {
    if (type === 'start') {
      setStartDate(value)
      if (onDateChange && endDate) {
        onDateChange(value, endDate)
      }
    } else {
      setEndDate(value)
      if (onDateChange && startDate) {
        onDateChange(startDate, value)
      }
    }
  }

  const handlePeriodPredict = async () => {
    if (!startDate || !endDate || startDate > endDate) {
      alert('올바른 기간을 선택해주세요.')
      return
    }
    
    // 최대 1주일(7일) 제한 검증
    const start = new Date(startDate)
    const end = new Date(endDate)
    const daysDiff = Math.ceil((end - start) / (1000 * 60 * 60 * 24))
    if (daysDiff > 7) {
      alert('예측 기간은 최대 7일(1주일)까지만 선택할 수 있습니다.')
      return
    }
    
    setPredicting(true)
    if (onPeriodPredict) {
      await onPeriodPredict(startDate, endDate)
    }
    setPredicting(false)
  }

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
                    onChange={(e) => {
                      const newStartDate = e.target.value
                      setStartDate(newStartDate)
                      // 시작일 변경 시 종료일이 7일을 초과하면 자동 조정
                      let newEndDate = endDate
                      if (newStartDate && endDate) {
                        const start = new Date(newStartDate)
                        const end = new Date(endDate)
                        const maxEndDate = new Date(start)
                        maxEndDate.setDate(maxEndDate.getDate() + 7)
                        if (end > maxEndDate) {
                          newEndDate = maxEndDate.toISOString().split('T')[0]
                          setEndDate(newEndDate)
                        }
                        // 종료일이 시작일보다 이전이면 조정
                        if (end < start) {
                          newEndDate = newStartDate
                          setEndDate(newEndDate)
                        }
                      }
                      // Dashboard에 날짜 변경 알림 (자동으로 예측 실행됨)
                      if (onDateChange && newStartDate) {
                        onDateChange(newStartDate, newEndDate || endDate)
                      }
                    }}
                    className="date-input"
                    min={new Date().toISOString().split('T')[0]}
                  />
                  <span className="date-separator">~</span>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => {
                      const newEndDate = e.target.value
                      // 최대 1주일 제한 검증
                      if (newEndDate && startDate) {
                        const start = new Date(startDate)
                        const end = new Date(newEndDate)
                        const daysDiff = Math.ceil((end - start) / (1000 * 60 * 60 * 24))
                        if (daysDiff > 7) {
                          alert('예측 기간은 최대 7일(1주일)까지만 선택할 수 있습니다.')
                          return
                        }
                        if (end < start) {
                          alert('종료일은 시작일 이후여야 합니다.')
                          return
                        }
                      }
                      setEndDate(newEndDate)
                      // Dashboard에 날짜 변경 알림 (자동으로 예측 실행됨)
                      if (onDateChange && newEndDate && startDate) {
                        onDateChange(startDate, newEndDate)
                      }
                    }}
                    className="date-input"
                    min={startDate}
                    max={startDate ? (() => {
                      const maxDate = new Date(startDate)
                      maxDate.setDate(maxDate.getDate() + 7)
                      return maxDate.toISOString().split('T')[0]
                    })() : undefined}
                  />
                </div>
                <div className="date-input-hint">최대 1주일만 선택 가능합니다</div>
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
                  <div className="stat-label">{dateLabel} 예상 방문자</div>
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
                  <span>{dateLabel} 방문 예측: 전일 대비 <strong>{parseFloat(yesterdayChange) > 0 ? '+' : ''}{yesterdayChange}%</strong></span>
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

