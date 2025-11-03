import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './TrendChart.css'

function TrendChart({ data, loading }) {
  if (loading) {
    return <div className="chart-loading">데이터 로딩 중...</div>
  }

  if (!data || !Array.isArray(data) || data.length === 0) {
    return <div className="chart-empty">트렌드 데이터가 없습니다.</div>
  }

  // 날짜 형식 변환
  const chartData = data.map(item => ({
    ...item,
    date: item.date ? new Date(item.date).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' }) : item.date,
    visits: item.visits || 0
  }))

  return (
    <div className="trend-chart">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis 
            dataKey="date"
            stroke="#64748b"
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            stroke="#64748b"
            tick={{ fontSize: 12 }}
            label={{ value: '방문 수', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            contentStyle={{
              background: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
            }}
            formatter={(value) => `${value.toLocaleString()}명`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="visits" 
            stroke="#667eea" 
            strokeWidth={3}
            name="방문 수"
            dot={{ r: 5, fill: '#667eea' }}
            activeDot={{ r: 8 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default TrendChart
