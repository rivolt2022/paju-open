import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import LoadingSpinner from './LoadingSpinner'
import './PredictionChart.css'

function PredictionChart({ data, loading, date = null }) {
  if (loading) {
    return (
      <div className="chart-loading">
        <LoadingSpinner message="차트 데이터 로딩 중..." size="medium" />
      </div>
    )
  }

  if (!data || !data.predictions || data.predictions.length === 0) {
    return <div className="chart-empty">예측 데이터가 없습니다.</div>
  }
  
  const dateLabel = date ? new Date(date).toLocaleDateString('ko-KR', { month: 'long', day: 'numeric', weekday: 'long' }) : '오늘'

  // 차트 데이터 형식 변환 (예측 vs 실제 비교)
  const chartData = data.predictions.map(pred => ({
    name: pred.space?.length > 10 ? pred.space.substring(0, 10) + '...' : pred.space || pred.spot || 'N/A',
    fullName: pred.space || pred.spot || 'N/A',
    예측방문: pred.predicted_visit || 0,
    실제방문: pred.actual_visit || pred.predicted_visit || 0,
    혼잡도: parseFloat(((pred.crowd_level || 0) * 100).toFixed(1)),
    오차율: pred.actual_visit ? Math.abs(((pred.predicted_visit - pred.actual_visit) / pred.actual_visit) * 100).toFixed(1) : 0,
  }))

  // 데이터 확인용 로그
  console.log('차트 데이터:', chartData)

  return (
    <div className="prediction-chart">
      <div className="chart-header">
        <span className="chart-date-label">{dateLabel} 예측 데이터</span>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="name" 
            angle={-45}
            textAnchor="end"
            height={100}
            interval={0}
            tick={{ fontSize: 11 }}
          />
          <YAxis 
            yAxisId="left" 
            label={{ value: '예측 방문 수', angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 11 }}
          />
          <YAxis 
            yAxisId="right" 
            orientation="right"
            label={{ value: '혼잡도 (%)', angle: 90, position: 'insideRight' }}
            domain={[0, 100]}
            tick={{ fontSize: 11 }}
          />
          <Tooltip 
            formatter={(value, name) => {
              if (name === '예측방문') {
                return [`${value.toLocaleString()}명`, '예측 방문']
              } else if (name === '혼잡도') {
                return [`${value}%`, '혼잡도']
              }
              return [value, name]
            }}
            labelFormatter={(label) => {
              const item = chartData.find(d => d.name === label || d.fullName === label)
              return item?.fullName || label
            }}
          />
          <Legend />
          <Bar 
            dataKey="예측방문" 
            fill="#667eea"
            yAxisId="left"
            name="예측 방문"
            radius={[4, 4, 0, 0]}
          />
          <Bar 
            dataKey="실제방문" 
            fill="#764ba2"
            yAxisId="left"
            name="실제 방문"
            radius={[4, 4, 0, 0]}
          />
          <Line 
            type="monotone" 
            dataKey="혼잡도" 
            stroke="#f59e0b" 
            strokeWidth={3}
            yAxisId="right"
            name="혼잡도 (%)"
            dot={{ r: 5 }}
            activeDot={{ r: 8 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

export default PredictionChart