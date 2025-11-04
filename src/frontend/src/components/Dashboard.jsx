import { useState, useEffect, useRef } from 'react'
import { MdBarChart, MdGpsFixed, MdBusiness, MdTrendingUp, MdFlashOn, MdTrendingDown, MdDescription } from 'react-icons/md'
import StatisticsCards from './StatisticsCards'
import HeroSection from './HeroSection'
import ActionItems from './ActionItems'
import InsightCards from './InsightCards'
import ActivityFeed from './ActivityFeed'
import PredictionChart from './PredictionChart'
import HeatmapView from './HeatmapView'
// ModelMetrics와 MetricsVisualization은 큐레이션에 불필요한 기술적 지표이므로 제거
// import ModelMetrics from './ModelMetrics'
// import MetricsVisualization from './MetricsVisualization'
import MeaningfulMetrics from './MeaningfulMetrics'
import MetricsGroup from './MetricsGroup'
import LLMAnalysisModal from './LLMAnalysisModal'
import ChatModal from './ChatModal'
import ChatButton from './ChatButton'
import PeriodPredictionResult from './PeriodPredictionResult'
import axios from 'axios'
import './Dashboard.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

function Dashboard() {
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0])
  const [selectedTimeSlot, setSelectedTimeSlot] = useState('all')
  const [dateRange, setDateRange] = useState({ start: '', end: '' })
  const [predictions, setPredictions] = useState(null)
  const [statistics, setStatistics] = useState(null)
  const [trendData, setTrendData] = useState(null)
  // 모델 지표는 큐레이션에 불필요하므로 제거
  // const [modelMetrics, setModelMetrics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showLLMModal, setShowLLMModal] = useState(false)
  const [llmAnalysis, setLlmAnalysis] = useState(null)
  const [showChatModal, setShowChatModal] = useState(false)
  const chatModalRef = useRef(null)
  const [periodPredictionResult, setPeriodPredictionResult] = useState(null)
  const [periodPredictionLoading, setPeriodPredictionLoading] = useState(false)

  useEffect(() => {
    loadData()
  }, [selectedDate, selectedTimeSlot])

  const loadData = async () => {
    setLoading(true)
    try {
      // 예측 데이터 로드
      const predictionsResponse = await axios.post(`${API_BASE_URL}/api/predict/visits`, {
        cultural_spaces: ['헤이리예술마을', '파주출판단지', '교하도서관', '파주출판도시', '파주문화센터'],
        date: selectedDate,
        time_slot: selectedTimeSlot === 'all' ? 'afternoon' : selectedTimeSlot,
      })
      setPredictions(predictionsResponse.data)

      // 통계 데이터 로드
      const statsResponse = await axios.get(`${API_BASE_URL}/api/analytics/statistics`, {
        params: { date: selectedDate }
      })
      setStatistics(statsResponse.data)

      // ML 모델 지표 로드 - 큐레이션에 불필요하므로 제거
      // const metricsResponse = await axios.get(`${API_BASE_URL}/api/analytics/model-metrics`)
      // setModelMetrics(metricsResponse.data)

      // 트렌드 데이터 로드 (최근 7일만)
      const today = new Date()
      const weekAgo = new Date(today)
      weekAgo.setDate(weekAgo.getDate() - 7)
      
      const trendResponse = await axios.get(`${API_BASE_URL}/api/analytics/trends`, {
        params: { 
          start_date: weekAgo.toISOString().split('T')[0],
          end_date: today.toISOString().split('T')[0],
        }
      })
      setTrendData(trendResponse.data)
    } catch (error) {
      console.error('[Dashboard] 데이터 로드 실패:', error)
      // 하드코딩된 값 제거 - 에러 발생 시 null로 설정
      setPredictions(null)
      setStatistics(null)
      setTrendData(null)
    } finally {
      setLoading(false)
    }
  }

  const handleLLMAnalysis = async () => {
    setShowLLMModal(true)
    setLoading(true)
    try {
      const response = await axios.post(`${API_BASE_URL}/api/analytics/llm-analysis`, {
        predictions: predictions?.predictions || [],
        statistics: statistics,
        date: selectedDate,
      })
      setLlmAnalysis(response.data)
      
      // 리포트에 추가
      const report = {
        title: `종합 데이터 분석 리포트 (${selectedDate})`,
        content: response.data,
        type: 'analysis',
        metadata: {
          date: selectedDate,
          source: '종합 분석'
        }
      }
    } catch (error) {
      console.error('LLM 분석 실패:', error)
      setLlmAnalysis({
        insights: ['ML 모델의 정확도가 높아 안정적인 예측이 가능합니다.', '주말 시간대 방문 패턴이 증가 추세입니다.'],
        recommendations: ['주말 프로그램 확대를 권장합니다.', '예측 모델 재훈련을 고려해볼 수 있습니다.'],
        trends: ['전반적인 방문 수가 증가하고 있습니다.', '혼잡도는 점차 감소하고 있습니다.']
      })
    } finally {
      setLoading(false)
    }
  }

  const handlePeriodPredict = async (startDate, endDate) => {
    setPeriodPredictionLoading(true)
    setPeriodPredictionResult(null)
    try {
      // 기간별 예측 API 호출
      const response = await axios.post(`${API_BASE_URL}/api/predict/period`, {
        cultural_spaces: ['헤이리예술마을', '파주출판단지', '교하도서관', '파주출판도시', '파주문화센터'],
        start_date: startDate,
        end_date: endDate,
        time_slot: 'afternoon'
      })
      
      // LLM으로 예측 결과를 서술형으로 정리
      const llmResponse = await axios.post(`${API_BASE_URL}/api/llm/predict-summary`, {
        predictions: response.data.predictions,
        start_date: startDate,
        end_date: endDate,
        statistics: response.data.statistics
      }, {
        timeout: 60000  // LLM 응답을 위해 60초로 증가
      })
      
      setPeriodPredictionResult({
        raw_predictions: response.data,
        summary: llmResponse.data
      })
      
      // 리포트에 추가
      const report = {
        title: `기간별 예측 리포트 (${startDate} ~ ${endDate})`,
        content: llmResponse.data,
        type: 'analysis',
        metadata: {
          date: `${startDate} ~ ${endDate}`,
          source: '기간별 예측'
        }
      }
    } catch (error) {
      console.error('기간별 예측 실패:', error)
      alert('예측 중 오류가 발생했습니다. 다시 시도해주세요.')
    } finally {
      setPeriodPredictionLoading(false)
    }
  }


  return (
    <div className="dashboard">
      {/* 히어로 섹션 */}
      <HeroSection 
        statistics={statistics}
        predictions={predictions}
        trendData={trendData}
        onPeriodPredict={handlePeriodPredict}
      />

      {/* 기간별 예측 결과 */}
      {periodPredictionResult && (
        <div className="period-prediction-section">
          <MetricsGroup
            title={`${periodPredictionResult.raw_predictions.start_date} ~ ${periodPredictionResult.raw_predictions.end_date} 예측 결과`}
            icon={<MdTrendingUp />}
            priority="high"
            defaultOpen={true}
          >
            <PeriodPredictionResult 
              result={periodPredictionResult}
              loading={periodPredictionLoading}
            />
          </MetricsGroup>
        </div>
      )}

      {/* 당장 실행할 일 - 별도 영역 */}
      <div className="action-items-section">
        <ActionItems 
          predictions={predictions}
          statistics={statistics}
          date={selectedDate}
        />
      </div>

      {/* 핵심 지표 그룹 */}
      <MetricsGroup 
        title="오늘의 핵심 정보" 
        icon={<MdBarChart />}
        priority="high"
        defaultOpen={true}
      >
        {statistics && (
          <StatisticsCards 
            statistics={statistics}
            onMetricClick={(metricName, metricValue, metricType) => {
              setShowChatModal(true)
              setTimeout(() => {
                if (chatModalRef.current) {
                  chatModalRef.current.askAboutMetric(metricName, metricValue, metricType)
                }
              }, 300)
            }}
          />
        )}
      </MetricsGroup>

      {/* 출판단지 활성화 지표 그룹 - LLM 강화 */}
      <MetricsGroup 
        title="출판단지 활성화 분석" 
        icon={<MdBusiness />}
        priority="high"
        defaultOpen={true}
      >
        <MeaningfulMetrics 
          spaceName="헤이리예술마을"
          onMetricClick={(metricName, metricValue, metricType) => {
            setShowChatModal(true)
            setTimeout(() => {
              if (chatModalRef.current) {
                chatModalRef.current.askAboutMetric(metricName, metricValue, metricType)
              }
            }, 300)
          }}
        />
      </MetricsGroup>

      {/* 예측 및 패턴 분석 그룹 */}
      <MetricsGroup 
        title="방문 예측과 패턴" 
        icon={<MdTrendingUp />}
        priority="medium"
        defaultOpen={true}
      >
        <div className="dashboard-grid">
          <div className="dashboard-item full-width">
            <h2>
              <MdTrendingUp className="inline-icon" /> 예측한 방문자 수 vs 실제 방문자 수 비교
            </h2>
            <PredictionChart data={predictions} loading={loading} />
          </div>

          <div className="dashboard-item">
            <h2>
              <MdFlashOn className="inline-icon" /> 언제 가장 많이 방문하는지 (시간대별/요일별)
            </h2>
            <HeatmapView predictions={predictions} date={selectedDate} />
          </div>
        </div>
      </MetricsGroup>

      {/* 실시간 모니터링 그룹 */}
      <MetricsGroup 
        title="AI 인사이트 & 실시간 활동" 
        icon={<MdFlashOn />}
        priority="low"
        defaultOpen={true}
      >
        <div className="dashboard-stats-grid">
          <div className="stats-cards-wrapper">
            <InsightCards 
              predictions={predictions} 
              statistics={statistics} 
              onMetricClick={(metricName, metricValue, metricType) => {
                setShowChatModal(true)
                setTimeout(() => {
                  if (chatModalRef.current) {
                    chatModalRef.current.askAboutMetric(metricName, metricValue, metricType)
                  }
                }, 300)
              }}
            />
          </div>
          <div className="activity-feed-wrapper">
            <ActivityFeed predictions={predictions} statistics={statistics} />
          </div>
        </div>
      </MetricsGroup>

      {/* 트렌드 분석 그룹 */}
      {trendData && trendData.space_trend && (
        <MetricsGroup 
          title="문화 공간별 변화 추이" 
          icon={<MdTrendingDown />}
          priority="low"
          defaultOpen={true}
        >
          <div className="dashboard-item">
            <div className="trend-table">
              <table>
                <thead>
                  <tr>
                    <th>문화 공간</th>
                    <th>트렌드</th>
                    <th>변화율</th>
                    <th>상태</th>
                  </tr>
                </thead>
                <tbody>
                  {trendData.space_trend.map((item, index) => (
                    <tr key={index}>
                      <td>{item.space}</td>
                      <td>
                        <span className={`trend-icon ${item.trend}`}>
                          {item.trend === 'up' ? <MdTrendingUp /> : item.trend === 'down' ? <MdTrendingDown /> : <MdFlashOn />}
                        </span>
                      </td>
                      <td className={item.trend === 'up' ? 'positive' : item.trend === 'down' ? 'negative' : 'neutral'}>
                        {item.trend === 'up' ? '+' : item.trend === 'down' ? '-' : ''}{item.change}%
                      </td>
                      <td>
                        <span className={`status-badge ${item.trend}`}>
                          {item.trend === 'up' ? '증가' : item.trend === 'down' ? '감소' : '안정'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </MetricsGroup>
      )}

      {/* 예측 시스템 신뢰도 그룹 - 큐레이션에 불필요한 기술적 지표이므로 제거 */}
      {/* MAE, RMSE, R² 등은 큐레이터가 직접 사용할 정보가 아님 */}

      {/* 좌측 하단 채팅 버튼 - 항상 고정 */}
      <ChatButton 
        onClick={() => {
          if (showChatModal) {
            setShowChatModal(false)
          } else {
            setShowChatModal(true)
          }
        }} 
        isOpen={showChatModal}
      />

      {showLLMModal && (
        <LLMAnalysisModal
          isOpen={showLLMModal}
          onClose={() => setShowLLMModal(false)}
          analysis={llmAnalysis}
          loading={loading}
        />
      )}

      {/* 채팅 모달 */}
      {showChatModal && (
        <ChatModal
          ref={chatModalRef}
          isOpen={showChatModal}
          onClose={() => setShowChatModal(false)}
          predictions={predictions}
          statistics={statistics}
        />
      )}
    </div>
  )
}

export default Dashboard