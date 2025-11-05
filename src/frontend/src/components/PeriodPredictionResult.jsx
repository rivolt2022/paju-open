import { MdTrendingUp, MdTrendingDown, MdPeople, MdBarChart, MdLocationCity, MdLightbulb } from 'react-icons/md'
import LoadingSpinner from './LoadingSpinner'
import './PeriodPredictionResult.css'

function PeriodPredictionResult({ result, loading }) {
  if (loading) {
    return (
      <div className="period-prediction-loading">
        <LoadingSpinner message="예측 결과를 생성하는 중..." size="medium" />
      </div>
    )
  }

  if (!result || !result.summary) {
    return null
  }

  const { raw_predictions, summary } = result
  
  // 날짜 레이블 생성
  const startDate = raw_predictions?.start_date || ''
  const endDate = raw_predictions?.end_date || ''
  const dateLabel = startDate && endDate 
    ? `${new Date(startDate).toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' })} ~ ${new Date(endDate).toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' })}`
    : ''

  return (
    <div className="period-prediction-result">
      {dateLabel && (
        <div className="prediction-date-header">
          <span className="prediction-date-label">예측 기간: {dateLabel}</span>
        </div>
      )}
      {/* LLM으로 정리된 서술형 결과 */}
      <div className="prediction-summary">
        <div className="summary-header">
          <MdLightbulb className="summary-icon" />
          <h3>예측 결과 요약</h3>
        </div>
        <div className="summary-content">
          {summary.summary && (
            <div className="summary-text" dangerouslySetInnerHTML={{ __html: summary.summary.replace(/\n/g, '<br/>') }} />
          )}
          {!summary.summary && summary.overview && (
            <p className="summary-text">{summary.overview}</p>
          )}
        </div>
      </div>

      {/* 주요 인사이트 */}
      {summary.insights && summary.insights.length > 0 && (
        <div className="prediction-insights">
          <h4 className="insights-title">주요 인사이트</h4>
          <ul className="insights-list">
            {summary.insights.map((insight, index) => (
              <li key={index} className="insight-item">
                <span className="insight-bullet">•</span>
                <span>{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 문화 공간별 상세 예측 */}
      {raw_predictions.predictions && raw_predictions.predictions.length > 0 && (
        <div className="prediction-details">
          <h4 className="details-title">문화 공간별 예측</h4>
          <div className="space-predictions">
            {raw_predictions.predictions.map((space, index) => (
              <div key={index} className="space-prediction-card">
                <div className="space-header">
                  <MdLocationCity className="space-icon" />
                  <h5 className="space-name">{space.space || space.space_name}</h5>
                </div>
                <div className="space-metrics">
                  <div className="metric">
                    <MdPeople className="metric-icon" />
                    <div className="metric-content">
                      <span className="metric-label">예상 방문 수</span>
                      <span className="metric-value">
                        {(space.total_visits || space.avg_visits || 0).toLocaleString()}명
                      </span>
                    </div>
                  </div>
                  <div className="metric">
                    <MdBarChart className="metric-icon" />
                    <div className="metric-content">
                      <span className="metric-label">평균 혼잡도</span>
                      <span className="metric-value">
                        {((space.avg_crowd_level || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
                {space.trend && (
                  <div className="space-trend">
                    {space.trend === 'up' ? (
                      <MdTrendingUp className="trend-icon trend-up" />
                    ) : space.trend === 'down' ? (
                      <MdTrendingDown className="trend-icon trend-down" />
                    ) : null}
                    <span className="trend-text">
                      {space.trend === 'up' ? '증가 추세' : space.trend === 'down' ? '감소 추세' : '안정적'}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 추천사항 */}
      {summary.recommendations && summary.recommendations.length > 0 && (
        <div className="prediction-recommendations">
          <h4 className="recommendations-title">추천 사항</h4>
          <ul className="recommendations-list">
            {summary.recommendations.map((rec, index) => (
              <li key={index} className="recommendation-item">
                <span className="recommendation-bullet">✓</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default PeriodPredictionResult

