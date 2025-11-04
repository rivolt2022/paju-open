import { useState, useRef, useEffect, forwardRef, useImperativeHandle } from 'react'
import { createPortal } from 'react-dom'
import { MdAutoAwesome, MdPerson, MdClose, MdBarChart, MdLightbulb, MdTrendingUp } from 'react-icons/md'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './ChatModal.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

const ChatModal = forwardRef(({ predictions, statistics, modelMetrics, isOpen, onClose }, ref) => {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    return () => setMounted(false)
  }, [])

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }
    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [isOpen])
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `안녕하세요! **PAJU Culture Lab AI 분석 어시스턴트**입니다.

저는 메인 대시보드의 **ML 분석 지표를 중심으로** 인사이트를 제공합니다.

## 현재 대시보드의 주요 ML 지표:

### 모델 성능 지표
- **모델 정확도 (R²)**: 매우 높은 예측 정확도
- **평균 절대 오차 (MAE)**: 예측 오차 분석
- **K-fold 교차 검증**: 모델 신뢰도 평가

### 문화 공간 활성화 지표
- **총 예측 방문 수**: 문화 공간별 방문 예측
- **평균 혼잡도**: 공간별 혼잡도 분석
- **활성화 점수**: 접근성, 관심도, 잠재력, 활용도

### 유의미한 분석 지표
- **주말/평일 패턴**: 방문 패턴 분석
- **성연령별 타겟팅**: 타겟 집단 분석
- **출판단지 활성화 지수**: 종합 활성화 지표

## 어떤 지표에 대해 알고 싶으신가요?

지표 카드를 클릭하시거나 아래와 같은 질문을 해보세요:
- "모델의 R² 점수는 무엇을 의미하나요?"
- "활성화 점수가 낮은 이유는 무엇인가요?"
- "출판단지 활성화를 위한 최적 전략은?"
- "주말 방문 패턴 분석 결과를 설명해주세요"`
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const messagesEndRef = useRef(null)
  const abortControllerRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent])

  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => {
        inputRef.current?.focus()
      }, 300)
    }
  }, [isOpen])

  // 외부에서 호출 가능한 함수들
  useImperativeHandle(ref, () => ({
    askAboutMetric: (metricName, metricData, metricType) => {
      const question = generateMetricQuestion(metricName, metricData, metricType)
      handleQuery(question, metricData)
    },
    askQuestion: (question, contextData = {}) => {
      handleQuery(question, contextData)
    }
  }))

  const generateMetricQuestion = (metricName, metricData, metricType) => {
    const questions = {
      'total_visits': `📊 총 예측 방문 수 ${metricData?.toLocaleString() || '0'}명에 대해 ML 분석 관점에서 상세히 분석해주세요. 이 수치의 의미, 출판단지 활성화에 미치는 영향, 그리고 모델이 이를 어떻게 예측했는지 설명해주세요.`,
      'avg_crowd_level': `📊 평균 혼잡도 ${(metricData * 100)?.toFixed(1) || '0'}%를 ML 모델 예측 관점에서 분석해주세요. 이 혼잡도 수준이 적정한지, 예측 모델이 이를 어떻게 활용하는지 설명해주세요.`,
      'model_accuracy': `📊 ML 모델 정확도 ${(metricData * 100)?.toFixed(1) || '0'}%에 대해 상세히 분석해주세요. 이 정확도가 높은 편인지, 모델의 신뢰성과 실제 예측에 어떤 의미인지, K-fold 교차 검증 결과와 함께 설명해주세요.`,
      'active_spaces': `📊 활성 문화 공간 ${metricData || '0'}개에 대해 ML 모델 관점에서 분석해주세요. 모델이 이 공간들의 방문 패턴을 어떻게 예측하는지, 각 공간의 활성화 지표는 무엇인지 설명해주세요.`,
      'activation_score': `📊 활성화 점수 ${metricData?.toFixed(1) || '0'}점을 ML 분석 관점에서 해석해주세요. 이 점수의 구성 요소(접근성, 관심도, 잠재력, 활용도)를 분석하고, 출판단지 활성화를 위한 데이터 기반 개선 방안을 제시해주세요.`,
      'r2_score': `📊 모델의 R² 점수 ${(metricData * 100)?.toFixed(2) || '0'}%에 대해 상세히 설명해주세요. 이 수치가 의미하는 바, 모델의 예측 신뢰성, 실제 비즈니스 의사결정에 어떤 영향을 미치는지 ML 분석 관점에서 분석해주세요.`,
      'mae': `📊 평균 절대 오차(MAE) ${metricData?.toFixed(2) || '0'}명에 대해 분석해주세요. 이 오차 수준이 허용 가능한지, 모델 성능 개선을 위해 어떤 특징 엔지니어링이나 하이퍼파라미터 조정이 필요한지 설명해주세요.`,
      'weekend_analysis': `📊 주말/평일 패턴 분석 결과를 ML 모델의 시간적 특징 관점에서 설명해주세요. 이 패턴이 모델 예측에 어떻게 반영되는지, 그리고 출판단지 활성화 전략에 어떤 시사점을 주는지 데이터 기반으로 제시해주세요.`,
      'demographic': `📊 성연령별 타겟팅 분석 결과를 데이터 기반 관점에서 설명해주세요. ML 모델이 이러한 인구통계학적 특징을 어떻게 활용하여 예측하는지, 문화 프로그램 기획에 어떤 인사이트를 제공하는지 분석해주세요.`,
      'vitality': `📊 출판단지 활성화 지수를 ML 지표 관점에서 종합 분석해주세요. 각 지표(인구활력, 소비활력, 생산활력)의 의미와 상관관계, 그리고 구체적인 활성화 방안을 데이터 기반으로 제시해주세요.`,
      'optimal_time': `📊 최적 방문 시간대 분석 결과를 ML 모델의 시간대별 예측 패턴 관점에서 설명해주세요. 모델이 이를 어떻게 학습했는지, 그리고 프로그램 기획에 어떻게 활용할 수 있는지 제안해주세요.`
    }
    
    return questions[metricType] || `📊 ${metricName}에 대해 ML 분석 관점에서 상세히 분석해주세요.`
  }

  const handleQuery = async (question, contextData = {}) => {
    if (loading) return

    setInput('')
    setStreamingContent('')
    
    // 사용자 메시지 추가
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setLoading(true)

    // 채팅창으로 스크롤
    setTimeout(() => {
      scrollToBottom()
    }, 100)

    // 이전 요청 취소
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    abortControllerRef.current = new AbortController()

    try {
      // 스트리밍 API 호출
      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: question,
          context: {
            predictions: predictions,
            statistics: statistics,
            model_metrics: modelMetrics,
            metric_context: contextData
          }
        }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) {
        throw new Error('스트리밍 응답 오류')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullContent = ''

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // 마지막 불완전한 라인 보관

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') {
              setLoading(false)
              // 스트리밍 완료 시 메시지에 추가
              if (fullContent) {
                setMessages(prev => [...prev, { role: 'assistant', content: fullContent }])
                setStreamingContent('')
              }
              setTimeout(() => scrollToBottom(), 100)
              return
            }

            try {
              const parsed = JSON.parse(data)
              if (parsed.content) {
                fullContent += parsed.content
                setStreamingContent(fullContent)
              }
            } catch (e) {
              // JSON 파싱 실패 시 무시
            }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        return
      }
      console.error('채팅 오류:', error)
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: '죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.' 
      }])
    } finally {
      setLoading(false)
      setStreamingContent('')
      abortControllerRef.current = null
    }
  }

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    await handleQuery(userMessage)
    setInput('')
  }

  const handleQuickQuestion = async (question) => {
    await handleQuery(question)
  }

  const quickQuestions = [
    '모델의 R² 점수 의미 설명',
    '활성화 점수 개선 방법',
    '주말 방문 패턴 인사이트',
    '출판단지 활성화 전략',
    'ML 지표 종합 분석'
  ]

  if (!isOpen) return null
  if (!mounted) return null

  const modalContent = (
    <div 
      className="chat-modal-overlay" 
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose()
        }
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="chat-modal-title"
    >
      <div 
        className="chat-modal" 
        onClick={(e) => e.stopPropagation()}
      >
        <div className="chat-modal-header">
          <div className="chat-header-content">
            <h3 className="chat-title" id="chat-modal-title">
              <MdAutoAwesome className="inline-icon" /> AI 분석 어시스턴트
            </h3>
            <p className="chat-subtitle">ML 분석 지표 중심의 인사이트를 제공합니다</p>
          </div>
          <button className="chat-modal-close" onClick={onClose}>
            <MdClose />
          </button>
        </div>

        <div className="chat-messages">
          {messages.map((message, index) => (
            <div key={index} className={`chat-message chat-message-${message.role}`}>
              <div className="message-avatar">
                {message.role === 'user' ? <MdPerson /> : <MdAutoAwesome />}
              </div>
              <div className="message-content">
                <div className="message-text">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          ))}
          {loading && streamingContent && (
            <div className="chat-message chat-message-assistant">
              <div className="message-avatar"><MdAutoAwesome /></div>
              <div className="message-content">
                <div className="message-text">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {streamingContent}
                  </ReactMarkdown>
                  <span className="streaming-cursor">▊</span>
                </div>
              </div>
            </div>
          )}
          {loading && !streamingContent && (
            <div className="chat-message chat-message-assistant">
              <div className="message-avatar"><MdAutoAwesome /></div>
              <div className="message-content">
                <div className="message-loading">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="quick-questions">
          <p className="quick-questions-label">빠른 질문:</p>
          <div className="quick-questions-buttons">
            {quickQuestions.map((question, index) => (
              <button
                key={index}
                className="quick-question-btn"
                onClick={() => handleQuickQuestion(question)}
                disabled={loading}
              >
                {question}
              </button>
            ))}
          </div>
        </div>

        <form className="chat-input-form" onSubmit={handleSend}>
          <input
            ref={inputRef}
            type="text"
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="ML 분석 지표에 대해 질문하세요..."
            disabled={loading}
          />
          <button type="submit" className="chat-send-btn" disabled={loading || !input.trim()}>
            {loading ? '⏳' : '📤'}
          </button>
        </form>
      </div>
    </div>
  )

  return createPortal(modalContent, document.body)
})

ChatModal.displayName = 'ChatModal'

export default ChatModal

