import { useState } from 'react'
import './HeatmapView.css'

function HeatmapView({ predictions, date }) {
  // 날짜 레이블 생성
  const dateLabel = date ? new Date(date).toLocaleDateString('ko-KR', { month: 'long', day: 'numeric', weekday: 'long' }) : '오늘'
  
  // 시간대별 데이터 생성 (예시)
  const timeSlots = ['09:00', '12:00', '15:00', '18:00', '21:00']
  const daysOfWeek = ['월', '화', '수', '목', '금', '토', '일']
  
  // 더미 데이터 생성 (실제로는 API에서 받아와야 함)
  const generateHeatmapData = () => {
    const data = {}
    daysOfWeek.forEach((day, dayIdx) => {
      timeSlots.forEach((time, timeIdx) => {
        const key = `${day}-${time}`
        // 랜덤한 혼잡도 생성 (0.2 ~ 0.9)
        data[key] = 0.2 + Math.random() * 0.7
      })
    })
    return data
  }

  const [heatmapData] = useState(generateHeatmapData())

  const getIntensity = (value) => {
    if (value < 0.4) return 'low'
    if (value < 0.7) return 'medium'
    return 'high'
  }

  const getIntensityLabel = (intensity) => {
    switch (intensity) {
      case 'low': return '여유'
      case 'medium': return '보통'
      case 'high': return '혼잡'
      default: return ''
    }
  }

  return (
    <div className="heatmap-view">
      <div className="heatmap-header-label">
        <span className="heatmap-date-label">{dateLabel} 시간대별/요일별 혼잡도</span>
      </div>
      <div className="heatmap-container">
        <div className="heatmap-header">
          <div className="heatmap-axis-label time-label">시간대</div>
          <div className="heatmap-grid">
            <div className="heatmap-row-header">
              {daysOfWeek.map((day) => (
                <div key={day} className="heatmap-day-header">{day}</div>
              ))}
            </div>
            {timeSlots.map((time) => (
              <div key={time} className="heatmap-row">
                <div className="heatmap-time-header">{time}</div>
                {daysOfWeek.map((day) => {
                  const key = `${day}-${time}`
                  const value = heatmapData[key] || 0
                  const intensity = getIntensity(value)
                  return (
                    <div
                      key={key}
                      className={`heatmap-cell heatmap-${intensity}`}
                      title={`${day} ${time}: ${(value * 100).toFixed(1)}%`}
                    >
                      <span className="heatmap-value">{(value * 100).toFixed(0)}%</span>
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div className="heatmap-legend">
        <div className="legend-item">
          <span className="legend-color heatmap-low"></span>
          <span>여유 (0-40%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color heatmap-medium"></span>
          <span>보통 (40-70%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color heatmap-high"></span>
          <span>혼잡 (70-100%)</span>
        </div>
      </div>
    </div>
  )
}

export default HeatmapView
