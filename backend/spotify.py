# backend/spotify.py

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import AsyncGenerator, Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import uvicorn
import os
import logging

# Import your logic
from database import AsyncSessionLocal, save_feedback
from spotify_client import generate_dynamic_recommendations
from spotify_client import MusicRecommender  # <- new clean import

# ------------------ Logger Setup ------------------
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------ FastAPI App Setup ------------------
app = FastAPI()

# Serve static files (frontend)
app.mount(
    "/frontend",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend")),
    name="frontend"
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/api")
async def api_root():
    return {"message": "Welcome to the Music Recommender API"}

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
    preview_url: Optional[str] = None
    external_url: Optional[str] = None
    features: Dict[str, float]

class ChatResponse(BaseModel):
    text: str
    recommendations: List[TrackRecommendation]
    analysis: Dict[str, Any]

# ------------------ API Endpoints ------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    try:
        user_message = request.message
        user_id = request.user_id

        recommender = MusicRecommender()
        result = await recommender.recommend_songs(db, user_id, user_message)

        recommendations = result.get("recommendations", [])
        analysis = result.get("analysis", {})

        return JSONResponse(content={
            "text": "Here's your personalized playlist based on your mood and activities:"
                    if recommendations else "Sorry, I couldn't find any music that fits your vibe 😔",
            "recommendations": jsonable_encoder(recommendations),
            "analysis": {
                "detected_mood": analysis.get("mood", "unknown"),
                "activities": analysis.get("activities", []),
                "genres": analysis.get("genres", []),
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )

# ------------------ Run App ------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
