import { useState, useRef, useEffect, useImperativeHandle, forwardRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './ChatPanel.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

const ChatPanel = forwardRef(({ predictions, statistics, modelMetrics }, ref) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: '안녕하세요! PAJU Culture Lab 큐레이션 어시스턴트입니다. 👋\n\n출판단지 활성화를 위한 프로그램 기획에 도움이 필요하시면 언제든 질문해주세요!\n\n예시 질문:\n- "주말 오후에 헤이리예술마을에서 어떤 프로그램이 좋을까요?"\n- "20-30대 여성을 위한 프로그램을 추천해주세요"\n- "평일 방문 활성화를 위한 프로그램 제안해주세요"\n- "출판단지 특성에 맞는 프로그램 아이디어를 알려주세요"'
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const messagesEndRef = useRef(null)
  const abortControllerRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent])

  // 외부에서 호출 가능한 함수들
  useImperativeHandle(ref, () => ({
    askAboutMetric: (metricName, metricData, metricType) => {
      const question = generateMetricQuestion(metricName, metricData, metricType)
      handleQuery(question, metricData)
    },
    askQuestion: (question, contextData = {}) => {
      handleQuery(question, contextData)
    },
    scrollToChat: () => {
      setTimeout(() => {
        const chatPanel = document.querySelector('.chat-panel')
        if (chatPanel) {
          chatPanel.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }
      }, 100)
    }
  }))

  const generateMetricQuestion = (metricName, metricData, metricType) => {
    const questions = {
      'total_visits': `오늘의 총 예측 방문 수 ${metricData?.toLocaleString() || '0'}명에 대해 분석해주세요. 이 수치가 의미하는 바와 출판단지 활성화에 미치는 영향을 설명해주세요.`,
      'avg_crowd_level': `평균 혼잡도 ${(metricData * 100)?.toFixed(1) || '0'}%에 대해 분석해주세요. 이 혼잡도 수준이 좋은지, 개선이 필요한지 판단해주세요.`,
      'model_accuracy': `ML 모델 정확도 ${(metricData * 100)?.toFixed(1) || '0'}%에 대해 상세히 분석해주세요. 이 정확도가 높은 편인지, 어떤 의미인지 설명해주세요.`,
      'active_spaces': `현재 활성 문화 공간이 ${metricData || '0'}개입니다. 이 숫자의 의미와 각 공간의 활용도를 분석해주세요.`,
      'activation_score': `활성화 점수가 ${metricData?.toFixed(1) || '0'}점입니다. 이 점수의 의미와 출판단지 활성화를 위한 개선 방안을 제시해주세요.`,
      'r2_score': `모델의 R² 점수가 ${(metricData * 100)?.toFixed(2) || '0'}%입니다. 이 수치의 의미와 모델의 신뢰성을 분석해주세요.`,
      'mae': `평균 절대 오차(MAE)가 ${metricData?.toFixed(2) || '0'}명입니다. 이 오차 수준이 허용 가능한지, 어떻게 개선할 수 있는지 설명해주세요.`,
      'weekend_analysis': `주말/평일 패턴 분석 결과를 바탕으로 출판단지 활성화 전략을 제시해주세요.`,
      'demographic': `성연령별 타겟팅 분석 결과를 바탕으로 문화 프로그램 기획 방안을 제안해주세요.`,
      'vitality': `출판단지 활성화 지수를 분석하고, 구체적인 활성화 방안을 제시해주세요.`
    }
    
    return questions[metricType] || `${metricName}에 대해 상세히 분석해주세요.`
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
    setInput(question)
    // 자동으로 전송되도록 이벤트 생성
    const fakeEvent = { preventDefault: () => {} }
    await handleSend(fakeEvent)
  }

  const quickQuestions = [
    '이번 주말에 추천할 프로그램은?',
    '가족 단위 방문객을 위한 프로그램',
    '파주출판단지에서 운영할 프로그램 추천',
    '저녁 시간대에 적합한 프로그램은?',
    '출판 관련 프로그램 아이디어 제안'
  ]

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <h3 className="chat-title">🤖 AI 분석 어시스턴트</h3>
        <p className="chat-subtitle">ML 데이터에 대해 자유롭게 질문하세요</p>
      </div>

      <div className="chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`chat-message chat-message-${message.role}`}>
            <div className="message-avatar">
              {message.role === 'user' ? '👤' : '🤖'}
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
            <div className="message-avatar">🤖</div>
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
            <div className="message-avatar">🤖</div>
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
          type="text"
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="프로그램 기획이나 큐레이션에 대해 질문하세요..."
          disabled={loading}
        />
        <button type="submit" className="chat-send-btn" disabled={loading || !input.trim()}>
          {loading ? '⏳' : '📤'}
        </button>
      </form>
    </div>
  )
})

ChatPanel.displayName = 'ChatPanel'

export default ChatPanel
