import { useState, useEffect, useRef } from 'react'
import { Map, MapMarker } from 'react-kakao-maps-sdk'
import { useKakaoLoader } from './KakaoMapLoader'
import './MapView.css'

// 파주시 문화 공간 위치 좌표
const CULTURAL_SPACES = {
  '헤이리예술마을': { lat: 37.7617, lng: 126.6800 },
  '파주출판단지': { lat: 37.7600, lng: 126.6900 },
  '교하도서관': { lat: 37.7500, lng: 126.6800 },
  '파주출판도시': { lat: 37.7550, lng: 126.6850 },
}

function MapView({ predictions }) {
  const [selectedMarker, setSelectedMarker] = useState(null)
  const [map, setMap] = useState(null)
  const infoWindowsRef = useRef({})
  const markersRef = useRef({})
  const center = { lat: 37.7600, lng: 126.6900 }
  const [level] = useState(3)
  const { loaded, error } = useKakaoLoader()

  // InfoWindow 생성 및 관리
  useEffect(() => {
    if (!map || !loaded || !window.kakao?.maps) return

    Object.entries(CULTURAL_SPACES).forEach(([name, position]) => {
      const prediction = predictions?.predictions?.find((p) => p.space === name)
      const isCrowded = prediction?.crowd_level && prediction.crowd_level > 0.7
      const visit = prediction?.predicted_visit || 0
      const crowd = prediction?.crowd_level ? (prediction.crowd_level * 100).toFixed(1) : '0'
      const optimalTime = prediction?.optimal_time || 'N/A'

      // 마커 위치
      const markerPosition = new window.kakao.maps.LatLng(position.lat, position.lng)

      // 마커가 없으면 생성
      if (!markersRef.current[name]) {
        const marker = new window.kakao.maps.Marker({
          position: markerPosition,
          clickable: true,
        })
        marker.setMap(map)
        markersRef.current[name] = marker

        // 마커 클릭 이벤트
        window.kakao.maps.event.addListener(marker, 'click', () => {
          // 다른 InfoWindow 닫기
          Object.values(infoWindowsRef.current).forEach(iw => iw.close())

          // 현재 InfoWindow 토글
          if (infoWindowsRef.current[name] && infoWindowsRef.current[name].getMap()) {
            infoWindowsRef.current[name].close()
            delete infoWindowsRef.current[name]
            setSelectedMarker(null)
          } else {
            // InfoWindow 내용
            const content = `
              <div style="padding:10px;min-width:200px;">
                <h3 style="margin:0 0 8px 0;font-size:16px;font-weight:bold;">${name}</h3>
                <p style="margin:4px 0;font-size:14px;">예측 방문: <strong>${visit.toLocaleString()}명</strong></p>
                <p style="margin:4px 0;font-size:14px;">혼잡도: <strong>${crowd}%</strong></p>
                <p style="margin:4px 0;font-size:14px;">최적 시간: <strong>${optimalTime}</strong></p>
                <div style="margin-top:8px;padding:4px 8px;background:${isCrowded ? '#ffebee' : '#e8f5e9'};color:${isCrowded ? '#c62828' : '#2e7d32'};border-radius:4px;font-size:12px;font-weight:bold;">
                  ${isCrowded ? '🔴 혼잡' : '🟢 여유'}
                </div>
              </div>
            `

            const infoWindow = new window.kakao.maps.InfoWindow({
              content: content,
              removable: true,
            })

            infoWindow.open(map, marker)
            infoWindowsRef.current[name] = infoWindow
            setSelectedMarker(name)

            // InfoWindow 닫기 이벤트
            window.kakao.maps.event.addListener(infoWindow, 'closeclick', () => {
              delete infoWindowsRef.current[name]
              setSelectedMarker(null)
            })
          }
        })
      }
    })

    return () => {
      // 정리
      Object.values(infoWindowsRef.current).forEach(iw => iw.close())
      Object.values(markersRef.current).forEach(marker => marker.setMap(null))
    }
  }, [map, loaded, predictions])

  // SDK 로드 중
  if (!loaded) {
    return (
      <div className="map-loading">
        <p>지도를 불러오는 중...</p>
        <p style={{ fontSize: '0.9em', color: '#666' }}>카카오맵 SDK를 로드하고 있습니다.</p>
      </div>
    )
  }

  // SDK 로드 오류
  if (error) {
    return (
      <div className="map-loading">
        <p style={{ color: '#d32f2f' }}>⚠️ 지도를 불러올 수 없습니다</p>
        <p style={{ fontSize: '0.9em', color: '#666' }}>{error}</p>
      </div>
    )
  }

  return (
    <div className="map-container">
      <Map
        center={center}
        style={{ width: '100%', height: '400px' }}
        level={level}
        onCreate={setMap}
      >
        {Object.entries(CULTURAL_SPACES).map(([name, position]) => {
          const prediction = predictions?.predictions?.find(
            (p) => p.space === name
          )
          const isCrowded = prediction?.crowd_level && prediction.crowd_level > 0.7

          return (
            <MapMarker
              key={name}
              position={position}
              clickable={true}
            >
              {isCrowded ? (
                <div style={{
                  padding: '8px',
                  background: '#ff4444',
                  color: 'white',
                  borderRadius: '8px',
                  fontSize: '12px',
                  fontWeight: 'bold',
                  whiteSpace: 'nowrap',
                }}>
                  🔴 {name}
                </div>
              ) : (
                <div style={{
                  padding: '8px',
                  background: '#44ff44',
                  color: 'white',
                  borderRadius: '8px',
                  fontSize: '12px',
                  fontWeight: 'bold',
                  whiteSpace: 'nowrap',
                }}>
                  🟢 {name}
                </div>
              )}
            </MapMarker>
          )
        })}
      </Map>
      
      <div className="map-legend">
        <div className="legend-item">
          <span className="legend-color" style={{ background: '#44ff44' }}></span>
          여유 (혼잡도 &lt; 0.7)
        </div>
        <div className="legend-item">
          <span className="legend-color" style={{ background: '#ff4444' }}></span>
          혼잡 (혼잡도 ≥ 0.7)
        </div>
      </div>
    </div>
  )
}

export default MapView