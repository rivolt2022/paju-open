import { useState, useEffect } from 'react'
import { MdMenuBook, MdDateRange, MdPlayArrow } from 'react-icons/md'
import './HeroSection.css'

export default function HeroSection({ statistics, predictions, onDateChange, onDatePredict, selectedDate = null, loading = false }) {

  // 단일 날짜만 사용 (하루만 예측)
  const [selectedDateValue, setSelectedDateValue] = useState(
    selectedDate || new Date().toISOString().split('T')[0]
  )
  
  // props가 변경되면 로컬 상태 업데이트
  useEffect(() => {
    if (selectedDate) {
      setSelectedDateValue(selectedDate)
    }
  }, [selectedDate])

  const handleDateChange = (value) => {
    setSelectedDateValue(value)
    if (onDateChange) {
      onDateChange(value)  // 단일 날짜만 전달
    }
  }

  const handlePredict = async () => {
    if (!selectedDateValue) {
      alert('날짜를 선택해주세요.')
      return
    }
    
    if (onDatePredict) {
      await onDatePredict(selectedDateValue)
    }
  }

  // loading prop을 사용하여 예측 진행 상태 표시
  const isPredicting = loading

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

          {/* 예측 날짜 섹션 */}
          <div className="hero-period-section">
            <div className="period-card">
              <div className="period-header">
                <MdDateRange className="period-header-icon" />
                <span className="period-header-title">예측 날짜</span>
              </div>
              <div className="period-content">
                <div className="date-inputs">
                  <input
                    type="date"
                    value={selectedDateValue}
                    onChange={(e) => {
                      const newDate = e.target.value
                      handleDateChange(newDate)
                    }}
                    className="date-input"
                    min={new Date().toISOString().split('T')[0]}
                  />
                </div>
                <button
                  className="predict-button"
                  onClick={handlePredict}
                  disabled={isPredicting}
                >
                  {isPredicting ? (
                    <>
                      <span className="spinner"></span>
                      <span>진행 중...</span>
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

        </div>
      </div>

    </div>
  )
}

