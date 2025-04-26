from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator, Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime
import logging
import uvicorn
import os
import json

from database import AsyncSessionLocal, save_feedback, get_user_conversation_history,get_db
from feedback_index import FeedbackLearner
from recommendation import MusicRecommender
from spotify_client import generate_dynamic_recommendations, SpotifyClient
from chat import UnifiedChatbot

# ------------------ Logger Setup ------------------
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------ FastAPI App Setup ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ OAuth Client ------------------
oauth_client = SpotifyClient(
    client_id="736bb144677e448dad56d2fe2ab70cd0",
    client_secret="d7beffe6e8d740deb7e1ddd9a111c88f",
    redirect_uri="http://127.0.0.1:8000/callback"
)

@app.get("/login")
def login():
    try:
        if os.path.exists("spotify_token.json"):
            return HTMLResponse("""
            <script>
            window.opener.localStorage.setItem('spotify_logged_in', 'true');
            window.opener.document.getElementById('loginNotice').style.display = 'none';
            window.opener.document.getElementById('createPlaylist').disabled = false;
            window.close();
            </script>
            <p>🎧 You're already logged in to Spotify. Ready to create your playlist!</p>
            """)
        else:
            auth_url = oauth_client.auth_manager.get_authorize_url()
            return RedirectResponse(auth_url)
    except Exception as e:
        return HTMLResponse(f"❌ Login redirect failed: {e}", status_code=500)

@app.get("/callback")
def callback(code: str):
    try:
        token_info = oauth_client.auth_manager.get_access_token(code)
        with open("spotify_token.json", "w") as f:
            json.dump(token_info, f)

        return HTMLResponse("""
        <script>
            window.opener.localStorage.setItem('spotify_logged_in', 'true');
            window.opener.document.getElementById('loginNotice').style.display = 'none';
            window.opener.document.getElementById('createPlaylist').disabled = false;
            window.close();
        </script>
        <p>✅ Logged in successfully! You can now return and create a playlist.</p>
        """)
    except Exception as e:
        return HTMLResponse(f"❌ Login failed: {e}", status_code=500)

# ------------------ Static File Serving ------------------
frontend_path = Path(__file__).parent / "../frontend"
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse(frontend_path / "index.html")

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

# ------------------ Models ------------------
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
    valence: Optional[float] = None

class ChatResponse(BaseModel):
    text: str
    recommendations: List[TrackRecommendation]
    analysis: Dict[str, Any]

class PlaylistRequest(BaseModel):
    user_id: str
    playlist_name: str
    tracks: List[Dict[str, Any]]

# ------------------ Endpoints ------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    try:
        logger.info(f"Received chat request from user {request.user_id}: {request.message}")
        
        user_message = request.message
        user_id = request.user_id

        # 🧠 Retrieve previous conversation history
        history = await get_user_conversation_history(db, user_id)
        context_messages = [msg.message for msg in history if not msg.is_bot]
        full_context = "\n".join(context_messages + [user_message])
        
        logger.info(f"Context for user {user_id}: {full_context}")

        # 🔍 Generate recommendations with full conversational context
        recommender = MusicRecommender()
        result = await recommender.recommend_songs(db, user_id, full_context)
        
        logger.info(f"Recommendations generated: {len(result.get('recommendations', []))} tracks")

        recommendations = result.get("recommendations", [])
        analysis = result.get("analysis", {})

        # 🧠 Construct intelligent response based on analysis
        mood = analysis.get("mood", "some mood")
        genres = analysis.get("genres", [])
        genre_part = f"{genres[0]} music" if genres else mood
        activities = analysis.get("activities", [])

        # 🪄 Dynamic reply
        if recommendations:
            if activities:
                activity_str = ", ".join(activities)
                reply = f"Here's a playlist for {genre_part} that suits your vibe: {activity_str}."
            elif genres:
                reply = f"You're into {genres[0]} right now — check this playlist out."
            else:
                reply = f"This should fit your {mood} mood. Here's your playlist."
        else:
            reply = "I couldn't find anything perfect, but I'll keep improving with your feedback."

        logger.info(f"Returning response to user {user_id}")
        
        return JSONResponse(content={
            "text": reply,
            "recommendations": jsonable_encoder(recommendations),
            "analysis": {
                "detected_mood": mood,
                "activities": activities,
                "genres": genres,
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Chat error for user {request.user_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )

@app.get("/api/default_recommendations")
async def get_default_recommendations():
    try:
        # Example: Get some popular tracks as defaults
        spotify = SpotifyClient()
        tracks = spotify.search_tracks("popular", limit=5)
        
        return {
            "recommendations": [
                {
                    "id": track["id"],
                    "name": track["name"],
                    "artists": [artist["name"] for artist in track["artists"]],
                    "preview_url": track["preview_url"],
                    "external_url": track["external_urls"]["spotify"],
                    "album": {"images": track["album"]["images"]}
                }
                for track in tracks
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/create_playlist")
async def create_playlist_endpoint(req: PlaylistRequest):
    try:
        if not os.path.exists("spotify_token.json"):
            raise HTTPException(status_code=401, detail="Spotify token not found. Please log in again.")

        with open("spotify_token.json", "r") as f:
            token_info = json.load(f)

        # Check and refresh token if expired
        if oauth_client.auth_manager.is_token_expired(token_info):
            token_info = oauth_client.auth_manager.refresh_access_token(token_info["refresh_token"])
            with open("spotify_token.json", "w") as f:
                json.dump(token_info, f)

        bot = UnifiedChatbot(token_info=token_info)
        url = bot.create_playlist_from_results(req.playlist_name, req.tracks)
        return {"playlist_url": url}

    except Exception as e:
        logger.error(f"Playlist creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def handle_feedback(feedback: FeedbackRequest, db: AsyncSession = Depends(get_db)):
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