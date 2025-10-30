from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Spot(BaseModel):
    spot_id: str
    name: str
    lat: float
    lng: float
    category: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class SpotResponse(BaseModel):
    spots: List[Spot]
    meta: Dict[str, Optional[str]] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    C: float = 0.0
    S: float = 0.0
    A: float = 0.0


class CourseNode(BaseModel):
    spot_id: str
    order: int


class Course(BaseModel):
    path: List[CourseNode]
    value: Optional[float] = None
    alt: Optional[List[CourseNode]] = None


class CourseResponse(BaseModel):
    theme: str
    time: Optional[str] = None
    courses: List[Course]


