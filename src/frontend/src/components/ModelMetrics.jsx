import { MdSmartToy, MdGpsFixed, MdRule, MdSquare, MdBarChart, MdScience, MdCalendarToday, MdTrendingUp, MdCheckCircle, MdWarning, MdCancel, MdLightbulb } from 'react-icons/md'
import './ModelMetrics.css'

function ModelMetrics({ metrics }) {
  if (!metrics) return null

  // k-fold 교차 검증 결과가 있는 경우
  const hasKFold = metrics.cv_r2_mean !== undefined

  // 모델 정확도를 더 쉽게 표현
  const getAccuracyMessage = (r2) => {
    const accuracy = (r2 || 0) * 100
    if (accuracy >= 95) return { text: '매우 높은 정확도', icon: <MdLightbulb />, level: 'excellent' }
    if (accuracy >= 90) return { text: '높은 정확도', icon: <MdCheckCircle />, level: 'high' }
    if (accuracy >= 80) return { text: '양호한 정확도', icon: <MdCheckCircle />, level: 'good' }
    if (accuracy >= 70) return { text: '보통 정확도', icon: <MdWarning />, level: 'medium' }
    return { text: '낮은 정확도', icon: <MdCancel />, level: 'low' }
  }

  const getErrorExplanation = (mae, avgValue) => {
    if (!mae || !avgValue) return '데이터 부족'
    const errorRate = (mae / avgValue) * 100
    if (errorRate < 2) return '오차가 거의 없어 매우 정확합니다 (날씨 예보 수준)'
    if (errorRate < 5) return '오차가 매우 적어 신뢰할 수 있습니다'
    if (errorRate < 10) return '오차가 적어 신뢰할 수 있습니다'
    if (errorRate < 20) return '오차가 보통 수준입니다'
    return '오차가 큰 편입니다. 개선이 필요합니다'
  }
  
  // 실제 학습 데이터를 기반으로 평균값 계산
  const avgVisitValue = metrics.target_range?.mean || 43484.5  // 학습 결과의 평균값 사용

  const r2Value = hasKFold ? metrics.cv_r2_mean : metrics.r2
  const maeValue = hasKFold ? metrics.cv_mae_mean : metrics.mae
  const rmseValue = hasKFold ? metrics.cv_rmse_mean : metrics.rmse
  const mapeValue = hasKFold ? (metrics.final_mape || metrics.cv_mape_mean) : metrics.mape

  const accuracyInfo = getAccuracyMessage(r2Value)

  const metricCards = [
    {
      title: '예측 정확도 (전체적인 맞춤율)',
      value: (r2Value * 100).toFixed(1),
      unit: '%',
      icon: <MdGpsFixed />,
      color: 'success',
      description: accuracyInfo.text,
      explanation: `예측이 전체적으로 ${(r2Value * 100).toFixed(1)}% 정도 정확합니다. ${(r2Value * 100) >= 90 ? '날씨 예보 수준으로 신뢰할 수 있습니다.' : (r2Value * 100) >= 80 ? '대부분의 경우 정확하게 예측합니다.' : '예측 정확도를 더 높이기 위해 개선 중입니다.'}`,
      threshold: { good: 0.95, warning: 0.90 },
      easyRead: true,
      variance: hasKFold ? metrics.cv_r2_std : undefined
    },
    {
      title: '평균 예측 오차 (예측과 실제의 차이)',
      value: maeValue?.toFixed(1) || '0',
      unit: '명',
      icon: <MdRule />,
      color: 'primary',
      description: getErrorExplanation(maeValue, avgVisitValue),
      explanation: `예측한 방문자 수와 실제 방문자 수의 차이가 평균 ${maeValue?.toFixed(0) || 0}명 정도입니다. 전체 평균 방문자 수(${avgVisitValue.toFixed(0)}명) 대비 ${((maeValue / avgVisitValue) * 100).toFixed(1)}%의 오차입니다. ${maeValue && maeValue < 1000 ? '매우 정확한 예측입니다 (날씨 예보 수준).' : maeValue && maeValue < 2000 ? '정확한 예측입니다.' : '보통 수준의 예측입니다.'}`,
      threshold: { good: 1000, warning: 2000 },
      easyRead: true,
      variance: hasKFold ? metrics.cv_mae_std : undefined
    },
    {
      title: '큰 오차 발생 시 차이 (극단적인 경우)',
      value: rmseValue?.toFixed(1) || '0',
      unit: '명',
      icon: <MdSquare />,
      color: 'secondary',
      description: '예측이 많이 틀렸을 때의 평균 차이',
      explanation: `예측이 많이 틀린 경우 평균 ${rmseValue?.toFixed(0) || 0}명 정도 차이가 납니다. ${rmseValue && rmseValue < 2000 ? '극단적인 오차도 작은 편입니다.' : '가끔 큰 오차가 발생할 수 있습니다.'}`,
      threshold: { good: 1500, warning: 3000 },
      variance: hasKFold ? metrics.cv_rmse_std : undefined
    },
    {
      title: '상대 오차율 (비율로 본 오차)',
      value: (() => {
        // MAPE는 이미 백분율이거나 소수점으로 올 수 있음
        if (!mapeValue) return '0'
        // 1보다 작으면 소수점 (0.0286 → 2.86%), 1보다 크면 이미 백분율 (2.86)
        const mapePercent = mapeValue < 1 ? (mapeValue * 100) : mapeValue
        return mapePercent.toFixed(2)
      })(),
      unit: '%',
      icon: <MdBarChart />,
      color: 'info',
      description: '전체 예측 중 오차가 차지하는 비율',
      explanation: (() => {
        if (!mapeValue) return 'MAPE 데이터가 없습니다.'
        const mapePercent = mapeValue < 1 ? (mapeValue * 100) : mapeValue
        let level = ''
        if (mapePercent < 3) level = '매우 작은 오차입니다 (날씨 예보 수준).'
        else if (mapePercent < 5) level = '매우 작은 오차입니다.'
        else if (mapePercent < 10) level = '작은 오차입니다.'
        else level = '보통 수준의 오차입니다.'
        return `예측 중 평균 ${mapePercent.toFixed(2)}% 정도의 오차가 있습니다. ${level}`
      })(),
      threshold: { good: 3, warning: 5 },
      easyRead: true
    },
    {
      title: '검증 방법 (시스템을 어떻게 테스트했나요?)',
      value: hasKFold ? `${metrics.cv_folds_used || 5}번 교차 검증` : '학습/테스트 분할 검증',
      unit: '',
      icon: <MdScience />,
      color: 'neutral',
      description: hasKFold 
        ? `${metrics.cv_folds_used || 5}개의 서로 다른 데이터셋으로 ${metrics.cv_folds_used || 5}번 테스트하여 안정성과 신뢰성 확인`
        : '학습용 데이터와 테스트용 데이터를 나누어 정확도 확인',
      explanation: hasKFold 
        ? `이 시스템은 ${metrics.cv_folds_used || 5}번의 서로 다른 데이터로 테스트했습니다. 매번 다른 데이터를 사용하여 검증했기 때문에 매우 신뢰할 수 있습니다.`
        : '데이터를 학습용과 테스트용으로 나누어 검증했습니다.'
    },
    {
      title: '시스템 마지막 업데이트 날짜',
      value: metrics.last_training_date || 'N/A',
      unit: '',
      icon: <MdCalendarToday />,
      color: 'neutral',
      description: '예측 시스템이 최신 데이터로 마지막으로 업데이트된 날짜',
      explanation: metrics.last_training_date 
        ? `이 시스템은 ${metrics.last_training_date}에 최신 데이터(${metrics.n_samples || 0}개 샘플, ${metrics.n_features || 0}개 특징)로 학습되었습니다.`
        : '학습 날짜 정보가 없습니다.'
    },
    {
      title: '모델 정보',
      value: metrics.model_type || 'Random Forest',
      unit: '',
      icon: <MdSmartToy />,
      color: 'neutral',
      description: '사용된 머신러닝 모델 유형',
      explanation: `${metrics.model_type || 'Random Forest'} 모델을 사용하여 ${metrics.n_features || 0}개의 특징으로 학습했습니다.`
    }
  ]

  // k-fold 결과가 있는 경우 추가 정보
  if (hasKFold && metrics.cv_folds) {
    metricCards.push({
      title: '여러 번 테스트한 결과 일관성',
      value: `${metrics.cv_folds.length}번 테스트`,
      unit: '',
      icon: <MdTrendingUp />,
      color: 'neutral',
      description: '다양한 데이터로 여러 번 테스트했을 때 성능이 일관적인지 확인',
      folds: metrics.cv_folds
    })
  }

  const getStatus = (metric) => {
    if (!metric.threshold) return 'neutral'
    const value = parseFloat(metric.value)
    if (value <= metric.threshold.good) return 'good'
    if (value <= metric.threshold.warning) return 'warning'
    return 'bad'
  }

  return (
    <div className="model-metrics">
      <div className="metrics-header">
        <h2 className="metrics-title">
          <MdSmartToy className="header-icon" />
          예측 시스템은 얼마나 정확한가요?
        </h2>
        <p className="metrics-subtitle">이 시스템이 날씨 예보처럼 신뢰할 수 있는지 확인해보세요</p>
        {accuracyInfo && (
          <div className="accuracy-summary">
            <span className="accuracy-icon">{accuracyInfo.icon}</span>
            <span className="accuracy-text">{accuracyInfo.text}</span>
            <span className="accuracy-value">{(r2Value * 100).toFixed(1)}%</span>
          </div>
        )}
      </div>
      <div className="metrics-grid">
        {metricCards.map((metric, index) => {
          const status = getStatus(metric)
          const isFoldCard = metric.folds !== undefined
          
          return (
            <div key={index} className={`metric-card metric-card-${metric.color} metric-${status}`}>
              <div className="metric-header">
                <span className="metric-icon">{metric.icon}</span>
                <span className="metric-title">{metric.title}</span>
              </div>
              <div className="metric-body">
                <div className="metric-value">
                  <span className="metric-number">{metric.value}</span>
                  <span className="metric-unit">{metric.unit}</span>
                  {metric.variance && (
                    <span className="metric-variance">±{metric.variance.toFixed(1)}</span>
                  )}
                </div>
                {metric.explanation && (
                  <p className="metric-explanation">{metric.explanation}</p>
                )}
                {metric.description && !metric.explanation && (
                  <p className="metric-description">{metric.description}</p>
                )}
                {isFoldCard && metric.folds && (
                  <div className="folds-detail">
                    <p className="folds-title">Fold별 상세 결과:</p>
                    {metric.folds.map((fold, idx) => (
                      <div key={idx} className="fold-item">
                        <span>Fold {fold.fold}:</span>
                        <span>R² = {fold.r2.toFixed(3)}</span>
                        <span>MAE = {fold.mae.toFixed(0)}명</span>
                      </div>
                    ))}
                  </div>
                )}
                {status !== 'neutral' && (
                  <div className={`metric-status metric-status-${status}`}>
                    <span className="status-icon">
                      {status === 'good' ? <MdCheckCircle /> : status === 'warning' ? <MdWarning /> : <MdCancel />}
                    </span>
                    <span className="status-text">
                      {status === 'good' ? '양호' : status === 'warning' ? '주의' : '개선 필요'}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default ModelMetrics
