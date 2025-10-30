import React from 'react'
import { MapView } from './components/MapView'
import { CourseCard } from './components/CourseCard'

export default function App() {
    return (
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', height: '100vh' }}>
            <MapView />
            <div style={{ overflow: 'auto', padding: 12 }}>
                <h2>추천 코스</h2>
                <CourseCard theme="family" />
                <CourseCard theme="couple" />
                <CourseCard theme="night" />
            </div>
        </div>
    )
}


