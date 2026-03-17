"""
FastAPI服务 - 英文评论错误检测
REST API for English Error Detection
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from production_error_detector import (
    EnglishErrorDetector,
    ErrorDetectionPipeline,
    SentenceAnalysis
)

app = FastAPI(
    title="English Error Detection API",
    description="检测英文评论中的拼写、语法和标点错误",
    version="1.0.0"
)

pipeline = ErrorDetectionPipeline(min_confidence=0.5)


class ErrorDetail(BaseModel):
    position: int = Field(..., description="错误位置（0-indexed）")
    token: str = Field(..., description="出错的单词")
    type: str = Field(..., description="错误类型（SPELLING/GRAMMAR/PUNCTUATION）")
    confidence: float = Field(..., description="置信度（0-1）")
    correction: Optional[str] = Field(None, description="建议的修正")
    explanation: Optional[str] = Field(None, description="错误说明")


class CommentAnalysis(BaseModel):
    sentence: str = Field(..., description="输入的句子")
    num_errors: int = Field(..., description="检测到的错误数")
    quality_score: float = Field(..., description="质量评分（0-1）")
    errors: List[ErrorDetail] = Field(default_factory=list, description="错误列表")


class CheckCommentRequest(BaseModel):
    comment: str = Field(..., min_length=1, max_length=1000, description="要检查的评论")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度阈值")


class CheckCommentsRequest(BaseModel):
    comments: List[str] = Field(..., min_items=1, max_items=100, description="评论列表")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度阈值")


class CheckCommentResponse(BaseModel):
    analysis: CommentAnalysis
    status: str = "success"


class CheckCommentsResponse(BaseModel):
    analyses: List[CommentAnalysis]
    total_comments: int
    total_errors: int
    avg_quality: float
    status: str = "success"


@app.post("/check", response_model=CheckCommentResponse, summary="检查单条评论")
async def check_single_comment(request: CheckCommentRequest):
    try:
        detector = EnglishErrorDetector(min_confidence=request.threshold)
        analysis = detector._analyze_sentence(request.comment)
        
        return CheckCommentResponse(
            analysis=CommentAnalysis(
                sentence=analysis.sentence,
                num_errors=analysis.error_count,
                quality_score=analysis.quality_score,
                errors=[e.to_dict() for e in analysis.errors]
            )
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing comment: {str(e)}")


@app.post("/check-batch", response_model=CheckCommentsResponse, summary="批量检查评论")
async def check_multiple_comments(request: CheckCommentsRequest):
    try:
        detector = EnglishErrorDetector(min_confidence=request.threshold)
        analyses = detector.detect(request.comments)
        
        total_errors = sum(a.error_count for a in analyses)
        avg_quality = sum(a.quality_score for a in analyses) / len(analyses) if analyses else 0
        
        return CheckCommentsResponse(
            analyses=[
                CommentAnalysis(
                    sentence=a.sentence,
                    num_errors=a.error_count,
                    quality_score=a.quality_score,
                    errors=[e.to_dict() for e in a.errors]
                ) for a in analyses
            ],
            total_comments=len(analyses),
            total_errors=total_errors,
            avg_quality=avg_quality
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing comments: {str(e)}")


@app.get("/health", summary="健康检查")
async def health_check():
    return {
        "status": "healthy",
        "service": "English Error Detection API",
        "version": "1.0.0"
    }


@app.get("/", summary="API说明")
async def root():
    return {
        "service": "English Error Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /check": "检查单条评论",
            "POST /check-batch": "批量检查评论 (最多100条)",
            "GET /health": "健康检查",
            "GET /docs": "交互式文档 (Swagger UI)"
        },
        "example": {
            "single": {
                "url": "/check",
                "method": "POST",
                "body": {
                    "comment": "Teh product is excellent",
                    "threshold": 0.5
                }
            },
            "batch": {
                "url": "/check-batch",
                "method": "POST",
                "body": {
                    "comments": [
                        "Teh product is excellent",
                        "I likes this product"
                    ],
                    "threshold": 0.5
                }
            }
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
