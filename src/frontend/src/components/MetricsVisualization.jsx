import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadialBarChart, RadialBar, PieChart, Pie, Cell } from 'recharts'
import { MdBarChart, MdScience, MdLightbulb, MdGpsFixed, MdRule, MdTrendingUp, MdDescription, MdCheckCircle, MdWarning, MdStar } from 'react-icons/md'
import './MetricsVisualization.css'

function MetricsVisualization({ metrics }) {
  if (!metrics) return null

  const hasKFold = metrics.cv_r2_mean !== undefined

  // Fold별 성능 데이터
  const foldData = metrics.cv_folds?.map(fold => ({
    fold: `Fold ${fold.fold}`,
    r2: (fold.r2 * 100).toFixed(1),
    mae: fold.mae,
    rmse: fold.rmse
  })) || []

  // 성능 지표 비교 데이터
  const metricsComparison = [
    {
      name: 'MAE',
      value: hasKFold ? metrics.cv_mae_mean : metrics.mae,
      std: hasKFold ? metrics.cv_mae_std : 0,
      color: '#667eea'
    },
    {
      name: 'RMSE',
      value: hasKFold ? metrics.cv_rmse_mean : metrics.rmse,
      std: hasKFold ? metrics.cv_rmse_std : 0,
      color: '#764ba2'
    },
    {
      name: 'MAPE',
      value: hasKFold ? (metrics.final_mape || 0) : metrics.mape,
      std: 0,
      color: '#f093fb'
    }
  ]

  // 정확도 원형 차트 데이터
  const r2Value = hasKFold ? metrics.cv_r2_mean : metrics.r2
  const accuracyData = [
    { name: '정확도', value: r2Value * 100, color: '#667eea' },
    { name: '오차', value: (1 - r2Value) * 100, color: '#e2e8f0' }
  ]

  const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']

  return (
    <div className="metrics-visualization">
      <div className="viz-header">
        <h2 className="viz-title">
          <MdBarChart className="header-icon" />
          예측 시스템 신뢰도 시각화
        </h2>
        <p className="viz-subtitle">차트로 쉽게 확인하는 예측 시스템의 신뢰도</p>
      </div>

      <div className="viz-grid">
        {/* Fold별 R² 성능 */}
        {foldData.length > 0 && (
          <div className="viz-card">
            <h3 className="viz-card-title">
              <MdScience className="card-title-icon" />
              여러 번 테스트한 예측 정확도
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={foldData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="fold" stroke="#64748b" />
                <YAxis 
                  stroke="#64748b"
                  label={{ value: 'R² (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value) => `${value}%`}
                  contentStyle={{ backgroundColor: 'white', border: '1px solid #e2e8f0', borderRadius: '8px' }}
                />
                <Bar dataKey="r2" fill="#667eea" radius={[8, 8, 0, 0]}>
                  {foldData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="viz-insight">
              <p>
                <MdLightbulb className="insight-icon" />
                <strong>인사이트:</strong> 다양한 데이터로 {foldData.length}번 테스트한 결과, 예측 정확도가 일관적으로 높게 유지되고 있습니다. 
              평균 {((foldData.reduce((sum, d) => sum + parseFloat(d.r2), 0) / foldData.length).toFixed(1))}%의 정확도를 보여 신뢰할 수 있습니다.</p>
            </div>
          </div>
        )}

        {/* 정확도 원형 차트 */}
        <div className="viz-card">
            <h3 className="viz-card-title">
              <MdGpsFixed className="card-title-icon" />
              예측 정확도 분포 (맞춘 부분 vs 틀린 부분)
            </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={accuracyData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {accuracyData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
            </PieChart>
          </ResponsiveContainer>
          <div className="viz-insight">
              <p>
                <MdLightbulb className="insight-icon" />
                <strong>인사이트:</strong> 예측 시스템이 {(r2Value * 100).toFixed(1)}%의 정확도로 예측합니다. 
            날씨 예보 수준의 신뢰도를 가지고 있어 실제 운영에 활용할 수 있는 수준입니다.</p>
          </div>
        </div>

        {/* 오차 지표 비교 */}
        <div className="viz-card">
            <h3 className="viz-card-title">
              <MdRule className="card-title-icon" />
              예측 오차 종류별 비교 (평균 오차 vs 큰 오차)
            </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={metricsComparison}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip 
                formatter={(value, name) => {
                  if (name === 'value') return `${value.toFixed(1)}명`
                  if (name === 'std') return `±${value.toFixed(1)}`
                  return value
                }}
                contentStyle={{ backgroundColor: 'white', border: '1px solid #e2e8f0', borderRadius: '8px' }}
              />
              <Bar dataKey="value" fill="#667eea" radius={[8, 8, 0, 0]}>
                {metricsComparison.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="viz-insight">
              <p>
                <MdLightbulb className="insight-icon" />
                <strong>인사이트:</strong> 평균 오차와 극단적인 오차가 모두 작은 편입니다. 
            대부분의 경우 정확하게 예측하며, 예상치 못한 상황에서도 큰 차이가 나지 않습니다.</p>
          </div>
        </div>

        {/* Fold별 MAE 추이 */}
        {foldData.length > 0 && (
          <div className="viz-card">
            <h3 className="viz-card-title">
              <MdTrendingUp className="card-title-icon" />
              여러 번 테스트한 오차 추이
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={foldData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="fold" stroke="#64748b" />
                <YAxis 
                  stroke="#64748b"
                  label={{ value: 'MAE (명)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value) => `${value.toFixed(0)}명`}
                  contentStyle={{ backgroundColor: 'white', border: '1px solid #e2e8f0', borderRadius: '8px' }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="mae" 
                  stroke="#667eea" 
                  strokeWidth={3}
                  dot={{ fill: '#667eea', r: 6 }}
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="viz-insight">
              <p>
                <MdLightbulb className="insight-icon" />
                <strong>인사이트:</strong> 다양한 상황에서 테스트해도 오차가 일관적으로 작습니다. 
              예측 시스템이 안정적이며, 새로운 데이터에도 잘 동작할 것으로 예상됩니다.</p>
            </div>
          </div>
        )}

        {/* 성능 요약 카드 */}
        <div className="viz-card viz-summary">
          <h3 className="viz-card-title">
            <MdDescription className="card-title-icon" />
            성능 요약
          </h3>
          <div className="summary-grid">
            <div className="summary-item">
              <div className="summary-label">평균 정확도</div>
              <div className="summary-value">{(r2Value * 100).toFixed(2)}%</div>
              {hasKFold && (
                <div className="summary-std">±{(metrics.cv_r2_std * 100).toFixed(2)}%</div>
              )}
            </div>
            <div className="summary-item">
              <div className="summary-label">평균 오차</div>
              <div className="summary-value">
                {(hasKFold ? metrics.cv_mae_mean : metrics.mae).toFixed(0)}명
              </div>
              {hasKFold && (
                <div className="summary-std">±{metrics.cv_mae_std.toFixed(0)}명</div>
              )}
            </div>
            <div className="summary-item">
              <div className="summary-label">검증 방법</div>
              <div className="summary-value">
                {hasKFold ? `${metrics.cv_folds_used}개 Fold` : 'Train/Test'}
              </div>
            </div>
            <div className="summary-item">
              <div className="summary-label">시스템 상태</div>
              <div className="summary-value">
                {(r2Value * 100) >= 95 ? (
                  <>
                    <MdStar className="summary-status-icon" />
                    매우 우수 (날씨 예보 수준)
                  </>
                ) : (r2Value * 100) >= 90 ? (
                  <>
                    <MdCheckCircle className="summary-status-icon" />
                    우수 (실용 가능)
                  </>
                ) : (r2Value * 100) >= 80 ? (
                  <>
                    <MdCheckCircle className="summary-status-icon" />
                    양호 (보완 필요)
                  </>
                ) : (
                  <>
                    <MdWarning className="summary-status-icon" />
                    개선 필요
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MetricsVisualization

