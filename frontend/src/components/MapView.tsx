import React, { useEffect, useMemo, useState } from 'react'
import { Map, MapMarker, Polyline } from 'react-kakao-maps-sdk'
import { api } from '../lib/api'

export const MapView: React.FC = () => {
    const [spots, setSpots] = useState<any[]>([])
    const [edges, setEdges] = useState<any[]>([])

    useEffect(() => {
        api.getSpots().then((r) => setSpots(r.spots || []))
        api.getGraph().then((e) => setEdges(e || []))
    }, [])

    const center = useMemo(() => {
        const s = spots.find((v) => v.lat && v.lng)
        return s ? { lat: s.lat, lng: s.lng } : { lat: 37.75, lng: 126.8 }
    }, [spots])

    const pathEdges = useMemo(() => {
        return edges
            .map((d: any) => {
                const a = spots.find((s) => s.spot_id === d.source)
                const b = spots.find((s) => s.spot_id === d.target)
                if (!a || !b || !a.lat || !a.lng || !b.lat || !b.lng) return null
                return { path: [{ lat: a.lat, lng: a.lng }, { lat: b.lat, lng: b.lng }], A: d.A || 0 }
            })
            .filter(Boolean) as { path: { lat: number; lng: number }[]; A: number }[]
    }, [edges, spots])

    return (
        <Map center={center} level={8} style={{ width: '100%', height: '100%' }}>
            {pathEdges.map((e, idx) => (
                <Polyline key={idx} path={[e.path]} strokeWeight={Math.max(2, e.A * 6)} strokeColor="#ff0080" strokeOpacity={0.6} />
            ))}
            {spots.filter(s => s.lat && s.lng).map((s) => (
                <MapMarker key={s.spot_id} position={{ lat: s.lat, lng: s.lng }} title={s.name} />
            ))}
        </Map>
    )
}


