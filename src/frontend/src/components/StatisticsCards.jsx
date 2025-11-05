import { MdPeople, MdBarChart, MdGpsFixed, MdBusiness } from 'react-icons/md'
import LoadingSpinner from './LoadingSpinner'
import './StatisticsCards.css'

function StatisticsCards({ statistics, onMetricClick, date = null, startDate = null, endDate = null }) {
  if (!statistics) {
    return (
      <div className="statistics-loading">
        <LoadingSpinner message="통계 데이터를 불러오는 중..." size="medium" />
      </div>
    )
  }

  // 날짜 포맷 함수 (단일 날짜만 사용)
  const formatDateLabel = (dateValue) => {
    if (!dateValue) return '오늘'
    return new Date(dateValue).toLocaleDateString('ko-KR', { month: 'long', day: 'numeric', weekday: 'long' })
  }

  const dateLabel = formatDateLabel(date || startDate)
  const isPeriod = false  // 하루만 선택하므로 항상 false

  const cards = [
    {
      title: isPeriod ? '기간 총 예측 방문 수' : '총 예측 방문 수',
      value: statistics.total_visits?.toLocaleString() || '0',
      rawValue: statistics.total_visits || 0,
      unit: '명',
      icon: <MdPeople />,
      color: 'primary',
      change: isPeriod && statistics.avg_daily_visits ? `일평균 ${statistics.avg_daily_visits.toLocaleString()}명` : '+5.2%',
      trend: 'up',
      metricType: 'total_visits',
      description: isPeriod 
        ? `${dateLabel} 전체 문화 공간에 예상되는 총 방문자 수 (${statistics.period_days || 0}일간)`
        : `${dateLabel} 전체 문화 공간에 예상되는 방문자 수`
    },
    {
      title: isPeriod ? '기간 평균 혼잡도' : '평균 혼잡도',
      value: (statistics.avg_crowd_level * 100).toFixed(1),
      rawValue: statistics.avg_crowd_level || 0,
      unit: '%',
      icon: <MdBarChart />,
      color: 'secondary',
      change: '-2.1%',
      trend: 'down',
      metricType: 'avg_crowd_level',
      description: isPeriod
        ? `${dateLabel} 전체 문화 공간의 평균 혼잡 정도`
        : '전체 문화 공간의 평균 혼잡 정도'
    },
    {
      title: '예측 신뢰도',
      value: (statistics.model_accuracy * 100).toFixed(1),
      rawValue: statistics.model_accuracy || 0,
      unit: '%',
      icon: <MdGpsFixed />,
      color: 'success',
      change: statistics.model_accuracy >= 0.99 ? '매우 높음' : statistics.model_accuracy >= 0.95 ? '높음' : '+1.5%',
      trend: statistics.model_accuracy >= 0.95 ? 'excellent' : 'up',
      metricType: 'model_accuracy',
      description: statistics.model_accuracy >= 0.99 
        ? '날씨 예보 수준으로 매우 신뢰할 수 있습니다'
        : statistics.model_accuracy >= 0.95
        ? '날씨 예보처럼 신뢰할 수 있는 수준입니다'
        : '신뢰할 수 있는 수준입니다'
    },
    {
      title: '활성 문화 공간',
      value: statistics.active_spaces,
      rawValue: statistics.active_spaces || 0,
      unit: '개',
      icon: <MdBusiness />,
      color: 'info',
      change: isPeriod && statistics.period_days ? `${statistics.period_days}일간 분석` : null,
      trend: 'stable',
      metricType: 'active_spaces',
      description: isPeriod
        ? `${statistics.period_days || 0}일간 모니터링한 문화 공간 수`
        : '현재 모니터링 중인 문화 공간 수'
    }
  ]

  const handleCardClick = (card) => {
    if (onMetricClick) {
      onMetricClick(card.title, card.rawValue, card.metricType)
    }
  }

  return (
    <div className="statistics-cards">
      {cards.map((card, index) => (
        <div 
          key={index} 
          className={`stat-card stat-card-${card.color} ${onMetricClick ? 'clickable' : ''}`}
          onClick={() => handleCardClick(card)}
          title={onMetricClick ? '클릭하여 이 지표에 대해 AI에게 물어보기' : ''}
        >
          <div className="stat-card-header">
            <span className="stat-icon">{card.icon}</span>
            <span className="stat-title">{card.title}</span>
          </div>
          <div className="stat-card-body">
            <div>
              <div className="stat-value">
                <span className="stat-number">{card.value}</span>
                <span className="stat-unit">{card.unit}</span>
              </div>
              {card.description && (
                <div className="stat-description">{card.description}</div>
              )}
            </div>
            {card.change && (
              <div className={`stat-change stat-change-${card.trend}`}>
                <span className="change-icon">
                  {card.trend === 'up' ? '↗' : card.trend === 'down' ? '↘' : '→'}
                </span>
                <span className="change-value">{card.change}</span>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

export default StatisticsCards
