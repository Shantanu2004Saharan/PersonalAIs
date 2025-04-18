from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import AsyncGenerator, Optional, List, Dict, Any
import logging
from sqlalchemy.ext.asyncio import AsyncSession
import uvicorn
from datetime import datetime
import logging.config
from fastapi.staticfiles import StaticFiles
from database import AsyncSessionLocal
from recommendation import generate_dynamic_recommendations
from feedback_learner import FeedbackLearner, save_feedback
from pathlib import Path
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from fastapi.responses import HTMLResponse

# ------------------ Logger Setup ------------------
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------ FastAPI App Setup ------------------
app = FastAPI()

# Serve static files (frontend)
frontend_path = Path(__file__).parent/ "frontend"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# ------------------ Database Dependency ------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            await session.close()

# ------------------ Pydantic Models ------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str

class FeedbackRequest(BaseModel):
    user_id: str
    track_id: str
    liked: bool

class TrackRecommendation(BaseModel):
    id: str
    name: str
    artists: List[str]
    preview_url: Optional[str]
    external_url: str
    features: Dict[str, float]

class ChatResponse(BaseModel):
    text: str
    recommendations: List[TrackRecommendation]
    analysis: Dict[str, Any]

# ------------------ Frontend Serving ------------------
'''@app.get("/")
async def serve_frontend(request: Request):
    # Don't interfere with FastAPI docs or OpenAPI routes
    if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
        return RedirectResponse(url=request.url.path)
    
    return FileResponse(frontend_path / "index.html")

'''
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse(frontend_path / "index.html")

@app.get("/api")
async def api_root():
    return {"message": "Welcome to the Music Recommender API"}

# ------------------ API Endpoints ------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    try:
        user_message = request.message
        user_id = request.user_id

        mood, recommendations = await generate_dynamic_recommendations(user_message)

        return JSONResponse(content={
            "text": f"I detected a {mood} mood. Here are some music recommendations:",
            "recommendations": recommendations[:12],
            "analysis": {
                "detected_mood": mood,
                "confidence": 0.9,
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )

@app.post("/api/feedback")
async def handle_feedback(
    feedback: FeedbackRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        await save_feedback(
            user_id=feedback.user_id,
            track_id=feedback.track_id,
            liked=feedback.liked
        )

        feedback_learner = FeedbackLearner(db)
        await feedback_learner.process_feedback(
            feedback.user_id,
            feedback.track_id,
            feedback.liked
        )

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ Run App ------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "level": "INFO",
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"]
            },
        }
    )
