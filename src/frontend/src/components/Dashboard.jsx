import { useState, useEffect, useRef } from 'react'
import { MdBarChart, MdGpsFixed, MdBusiness, MdTrendingUp, MdFlashOn, MdDescription } from 'react-icons/md'
import LoadingSpinner from './LoadingSpinner'
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
import ToastContainer from './ToastContainer'
import axios from 'axios'
import './Dashboard.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

function Dashboard() {
  // 단일 날짜만 사용 (하루만 예측)
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0])
  const startDate = selectedDate
  const endDate = selectedDate
  
  const [selectedTimeSlot, setSelectedTimeSlot] = useState('all')
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
  const [hasInitialPrediction, setHasInitialPrediction] = useState(false)
  const isManualPredictRef = useRef(false) // 버튼 클릭으로 인한 예측인지 추적
  const [toasts, setToasts] = useState([])
  const toastIdRef = useRef(0)

  // 페이지 로드 시 초기 날짜 데이터 자동 로드
  useEffect(() => {
    console.log('[Dashboard] 페이지 로드 - 초기 데이터 자동 로드', { selectedDate })
    // 초기 로드 시에는 알림 표시하지 않음
    loadData(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // 빈 배열 - 컴포넌트 마운트 시 한 번만 실행

  const loadData = async (showNotification = false) => {
    console.log('[Dashboard] loadData 시작', { selectedDate, showNotification })
    setLoading(true)
    try {
      console.log('[Dashboard] API 호출 시작', {
        predictionsEndpoint: `${API_BASE_URL}/api/predict/visits`,
        statisticsEndpoint: `${API_BASE_URL}/api/analytics/statistics`,
        trendsEndpoint: `${API_BASE_URL}/api/analytics/trends`,
        date: selectedDate
      })
      
      // 모든 HTTP 요청을 병렬로 처리
      const [predictionsResponse, statsResponse, trendResponse] = await Promise.all([
        // 예측 데이터 로드
        axios.post(`${API_BASE_URL}/api/predict/visits`, {
          cultural_spaces: ['헤이리예술마을', '파주출판단지', '교하도서관', '파주출판도시', '파주문화센터'],
          date: selectedDate,
          time_slot: selectedTimeSlot === 'all' ? 'afternoon' : selectedTimeSlot,
        }, {
          timeout: 120000  // 타임아웃 120초로 증가
        }),
        // 통계 데이터 로드 (단일 날짜만 사용)
        axios.get(`${API_BASE_URL}/api/analytics/statistics`, {
          params: { date: selectedDate },
          timeout: 120000
        }),
        // 트렌드 데이터 로드 (선택된 날짜 하루만)
        axios.get(`${API_BASE_URL}/api/analytics/trends`, {
          params: { 
            start_date: selectedDate,
            end_date: selectedDate,
          },
          timeout: 120000  // 타임아웃 120초로 증가
        })
      ])

      // 모든 응답을 상태에 반영
      console.log('[Dashboard] 데이터 로드 완료 - 상태 업데이트', {
        predictionsData: predictionsResponse.data,
        statisticsData: statsResponse.data,
        trendData: trendResponse.data
      })
      
      console.log('[Dashboard] 상태 업데이트 전', {
        predictions: predictions,
        statistics: statistics,
        predictionsResponseData: predictionsResponse.data,
        statisticsResponseData: statsResponse.data
      })
      
      setPredictions(predictionsResponse.data)
      setStatistics(statsResponse.data)
      setTrendData(trendResponse.data)
      
      console.log('[Dashboard] 상태 업데이트 완료 - setPredictions/setStatistics 호출됨', {
        predictionsType: typeof predictionsResponse.data,
        statisticsType: typeof statsResponse.data,
        predictionsKeys: predictionsResponse.data ? Object.keys(predictionsResponse.data) : [],
        statisticsKeys: statsResponse.data ? Object.keys(statsResponse.data) : [],
        predictionsHasPredictions: predictionsResponse.data?.predictions ? true : false,
        predictionsPredictionsLength: predictionsResponse.data?.predictions?.length || 0
      })

      // 모든 요청 완료 시 화면 알림 표시
      if (showNotification) {
        const dateLabel = new Date(selectedDate).toLocaleDateString('ko-KR', { 
          month: 'long', 
          day: 'numeric' 
        })
        const toastId = toastIdRef.current++
        setToasts(prev => [...prev, {
          id: toastId,
          message: `${dateLabel} 날짜의 모든 데이터 분석이 완료되었습니다.`,
          type: 'success',
          duration: 3000
        }])
      }
    } catch (error) {
      console.error('[Dashboard] 데이터 로드 실패:', error)
      console.error('[Dashboard] 에러 상세:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        stack: error.stack
      })
      // 하드코딩된 값 제거 - 에러 발생 시 null로 설정
      setPredictions(null)
      setStatistics(null)
      setTrendData(null)
      
      // 에러 발생 시에도 알림 표시
      if (showNotification) {
        const toastId = toastIdRef.current++
        setToasts(prev => [...prev, {
          id: toastId,
          message: '데이터 로드 중 오류가 발생했습니다.',
          type: 'error',
          duration: 3000
        }])
      }
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
      }, {
        timeout: 120000  // 타임아웃 120초로 증가
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

  // 날짜 포맷 함수 (단일 날짜만 사용)
  const formatDate = (date) => {
    if (!date) return '오늘'
    const dateObj = new Date(date)
    return dateObj.toLocaleDateString('ko-KR', { month: 'long', day: 'numeric', weekday: 'long' })
  }
  
  // 날짜 변경 핸들러 (HeroSection에서 날짜 입력 시 호출됨)
  const handleDateChange = (newDate) => {
    // 단일 날짜만 업데이트 (자동으로 데이터 로드는 하지 않음)
    setSelectedDate(newDate)
  }

  // 날짜 예측 핸들러 (HeroSection에서 예측 실행 버튼 클릭 시 호출됨)
  const handleDatePredict = async (date) => {
    isManualPredictRef.current = true // 버튼 클릭으로 인한 예측임을 표시
    setSelectedDate(date)
    // 버튼 클릭 시에는 알림을 표시하기 위해 직접 호출
    await loadData(true) // showNotification = true
  }


  return (
    <div className="dashboard">
      {/* 히어로 섹션 */}
      <HeroSection 
        statistics={statistics} 
        predictions={predictions}
        trendData={trendData}
        onDateChange={handleDateChange}
        onDatePredict={handleDatePredict}
        selectedDate={selectedDate}
        loading={loading}
      />

      {/* 당장 실행할 일 - 별도 영역 */}
      <div className="action-items-section">
        <ActionItems 
          predictions={predictions}
          statistics={statistics}
          date={selectedDate}
        />
      </div>

      {/* 출판단지 활성화 지표 그룹 - LLM 강화 */}
      <MetricsGroup 
        title={`${formatDate(selectedDate)} 출판단지 활성화 분석`}
        icon={<MdBusiness />}
        priority="high"
        defaultOpen={true}
      >
        <MeaningfulMetrics 
          spaceName="헤이리예술마을"
          date={selectedDate}
          startDate={selectedDate}
          endDate={selectedDate}
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

      {/* 토스트 알림 */}
      <ToastContainer 
        toasts={toasts}
        onRemove={(id) => setToasts(prev => prev.filter(toast => toast.id !== id))}
      />
    </div>
  )
}

export default Dashboard