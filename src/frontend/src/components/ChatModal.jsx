import { useState, useRef, useEffect, forwardRef, useImperativeHandle } from 'react'
import { createPortal } from 'react-dom'
import { MdAutoAwesome, MdPerson, MdClose, MdBarChart, MdLightbulb, MdTrendingUp } from 'react-icons/md'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './ChatModal.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

const ChatModal = forwardRef(({ predictions, statistics, isOpen, onClose }, ref) => {
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
      content: `안녕하세요! **PAJU Culture Lab 큐레이션 어시스턴트**입니다.

저는 출판단지 활성화를 위한 **실질적인 큐레이션 제안**을 도와드립니다.

## 큐레이션 지원 기능:

✨ **프로그램 기획 도움**
- 어떤 프로그램을 언제 어디서 운영하면 좋을지
- 타겟 고객층에 맞는 프로그램 추천
- 시간대별 최적 프로그램 제안

📅 **일정 및 장소 추천**
- 주말/평일별 최적 운영 시간
- 문화 공간별 추천 프로그램
- 계절별 프로그램 추천

🎯 **실행 가능한 제안**
- 구체적인 프로그램 아이디어
- 운영 시 주의사항
- 효과적인 마케팅 시점

## 어떤 도움이 필요하신가요?

지표 카드를 클릭하시거나 아래와 같은 질문을 해보세요:
- "주말 오후에 헤이리예술마을에서 어떤 프로그램이 좋을까요?"
- "20-30대 여성을 위한 프로그램을 추천해주세요"
- "이번 주말에 가장 효과적인 프로그램은?"
- "출판단지 활성화를 위한 프로그램 제안해주세요"
- "혼잡도가 낮은 시간대에 운영하면 좋을 프로그램은?"`
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
    // 큐레이션 중심 질문 - ML 지표 설명이 아닌 실질적 제안 요청
    const questions = {
      'total_visits': `예상 방문 수가 ${metricData?.toLocaleString() || '0'}명으로 예측됩니다. 이 예측을 바탕으로 어떤 프로그램을 운영하면 좋을까요? 구체적인 프로그램 아이디어와 운영 시점을 제안해주세요.`,
      'avg_crowd_level': `평균 혼잡도가 ${(metricData * 100)?.toFixed(1) || '0'}%입니다. 이 혼잡도 수준을 고려해서 어떤 프로그램을 추천할 수 있을까요? 혼잡도가 높은 시간대와 낮은 시간대에 각각 어떤 프로그램이 적합할지 제안해주세요.`,
      'model_accuracy': `예측 신뢰도가 ${(metricData * 100)?.toFixed(1) || '0'}%입니다. 이 예측을 신뢰하고 어떤 프로그램을 기획하면 좋을까요? 구체적인 프로그램 제안과 운영 계획을 알려주세요.`,
      'active_spaces': `활성 문화 공간이 ${metricData || '0'}개입니다. 각 공간의 특성을 고려해서 어떤 프로그램을 운영하면 좋을까요? 공간별 맞춤 프로그램을 추천해주세요.`,
      'activation_score': `활성화 점수가 ${metricData?.toFixed(1) || '0'}점입니다. 이 점수를 높이기 위해 어떤 프로그램을 기획하면 좋을까요? 접근성, 관심도, 잠재력, 활용도를 모두 고려한 구체적인 프로그램 제안을 해주세요.`,
      'r2_score': `예측 정확도가 ${(metricData * 100)?.toFixed(2) || '0'}%입니다. 이 정확한 예측을 바탕으로 어떤 프로그램을 추천할 수 있을까요?`,
      'weekend_analysis': `주말/평일 패턴 분석 결과를 바탕으로 주말과 평일에 각각 어떤 프로그램을 운영하면 좋을까요? 구체적인 프로그램 아이디어와 운영 시간대를 제안해주세요.`,
      'demographic': `성연령별 타겟팅 분석 결과를 바탕으로 어떤 타겟 집단을 위한 프로그램을 기획하면 좋을까요? 구체적인 프로그램 아이디어와 마케팅 방안을 제안해주세요.`,
      'vitality': `출판단지 활성화 지수를 고려해서 어떤 프로그램을 운영하면 출판단지 활성화에 도움이 될까요? 구체적인 프로그램 제안과 기대 효과를 알려주세요.`,
      'optimal_time': `최적 방문 시간대 분석 결과를 바탕으로 이 시간대에 어떤 프로그램을 운영하면 좋을까요? 구체적인 프로그램 아이디어와 운영 방안을 제안해주세요.`
    }
    
    return questions[metricType] || `이 ${metricName} 정보를 바탕으로 어떤 프로그램을 추천할 수 있을까요? 구체적인 큐레이션 제안을 해주세요.`
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
            // model_metrics 제거 - 큐레이션에 불필요한 기술적 지표
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
    '주말 오후에 어떤 프로그램이 좋을까요?',
    '20-30대 여성을 위한 프로그램 추천',
    '헤이리예술마을에서 운영할 최적 프로그램은?',
    '평일 방문 활성화를 위한 프로그램 제안',
    '출판단지 특성에 맞는 프로그램 아이디어'
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
            <p className="chat-subtitle">출판단지 활성화를 위한 큐레이션 제안을 제공합니다</p>
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
            placeholder="프로그램 기획이나 큐레이션에 대해 질문하세요..."
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

