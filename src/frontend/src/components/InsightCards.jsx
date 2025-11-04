import { useState, useEffect } from 'react'
import { MdRocketLaunch, MdAccessTime, MdGpsFixed, MdBarChart, MdLightbulb } from 'react-icons/md'
import './InsightCards.css'

function InsightCards({ predictions, statistics, onMetricClick }) {
  const [activeCard, setActiveCard] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveCard(prev => (prev + 1) % 4)
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const insightsList = [
    {
      icon: <MdRocketLaunch />,
      title: '최고 활성 공간',
      value: predictions?.predictions?.[0]?.space || '헤이리예술마을',
      description: '오늘 가장 많은 방문자가 예상되는 문화 공간',
      color: 'primary',
      trend: 'up',
      change: '+8.5%'
    },
    {
      icon: <MdAccessTime />,
      title: '최적 방문 시간',
      value: predictions?.predictions?.[0]?.optimal_time || '15:00-17:00',
      description: '혼잡도가 낮고 방문하기 가장 좋은 시간대',
      color: 'success',
      trend: 'stable'
    },
    {
      icon: <MdGpsFixed />,
      title: '예측 신뢰도',
      value: statistics?.model_accuracy 
        ? `${(statistics.model_accuracy * 100).toFixed(1)}%`
        : '95.0%',
      description: (() => {
        const accuracy = statistics?.model_accuracy ? (statistics.model_accuracy * 100) : 95
        if (accuracy >= 99) return '날씨 예보 수준으로 매우 신뢰할 수 있는 예측입니다'
        if (accuracy >= 95) return '날씨 예보처럼 신뢰할 수 있는 예측입니다'
        return '신뢰할 수 있는 예측입니다'
      })(),
      color: 'info',
      trend: 'up',
      change: null
    },
    {
      icon: <MdBarChart />,
      title: '평균 혼잡도',
      value: statistics?.avg_crowd_level 
        ? `${(statistics.avg_crowd_level * 100).toFixed(1)}%`
        : '40.0%',
      description: '전체 문화 공간의 평균 혼잡 수준',
      color: 'warning',
      trend: 'down',
      change: '-2.1%'
    }
  ]

  const getTopSpaces = () => {
    if (!predictions?.predictions) return []
    return [...predictions.predictions]
      .sort((a, b) => (b.predicted_visit || 0) - (a.predicted_visit || 0))
      .slice(0, 3)
  }

  return (
    <div className="insight-cards-section">
      <div className="section-header">
        <h2 className="section-title">
          <MdLightbulb className="title-icon" />
          AI 인사이트
        </h2>
        <p className="section-subtitle">실시간 데이터 분석 기반 인사이트</p>
      </div>

      <div className="insight-cards-grid">
        {insightsList.map((insight, index) => (
        <div
          key={index}
          className={`insight-card insight-card-${insight.color} ${activeCard === index ? 'active' : ''} ${onMetricClick ? 'clickable' : ''}`}
          onClick={() => {
            setActiveCard(index)
            if (onMetricClick) {
              const metricType = insight.title === '최고 활성 공간' ? 'active_spaces' :
                                 insight.title === '최적 방문 시간' ? 'optimal_time' :
                                 insight.title === '예측 정확도' ? 'r2_score' :
                                 insight.title === '평균 혼잡도' ? 'avg_crowd_level' : 'general'
              
              let metricValue = insight.value
              if (insight.title === '예측 신뢰도') {
                metricValue = statistics?.model_accuracy || 0.95
              } else if (insight.title === '평균 혼잡도') {
                metricValue = statistics?.avg_crowd_level || 0.4
              }
              
              onMetricClick(insight.title, metricValue, metricType)
            }
          }}
          title={onMetricClick ? '클릭하여 이 인사이트에 대해 AI에게 물어보기' : ''}
        >
            <div className="card-glow"></div>
            <div className="card-content">
              <div className="card-header">
                <div className="card-icon">{insight.icon}</div>
                <div className="card-title-section">
                  <h3 className="card-title">{insight.title}</h3>
                  {insight.change && (
                    <span className={`card-trend card-trend-${insight.trend}`}>
                      {insight.trend === 'up' ? '↗' : insight.trend === 'down' ? '↘' : '→'}
                      {insight.change}
                    </span>
                  )}
                </div>
              </div>
              <div className="card-body">
                <div className="card-value">{insight.value}</div>
                <p className="card-description">{insight.description}</p>
              </div>
              {insight.color === 'primary' && getTopSpaces().length > 0 && (
                <div className="card-footer">
                  <div className="top-spaces">
                    {getTopSpaces().map((space, idx) => (
                      <div key={idx} className="top-space-item">
                        <span className="space-rank">{idx + 1}</span>
                        <span className="space-name">{space.space}</span>
                        <span className="space-visits">
                          {space.predicted_visit?.toLocaleString()}명
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default InsightCards

