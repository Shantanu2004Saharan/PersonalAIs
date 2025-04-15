from typing import Dict, List
import logging
from database import save_user_message, get_user_conversation_history
from recommendation import MusicRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self):
        self.recommender = MusicRecommender()
        self.responses = {
            "welcome": "Hi! I'm your music recommendation assistant. Tell me how you're feeling or what kind of music you'd like to hear.",
            "no_input": "I didn't quite get that. Could you describe your mood or what you're doing?",
            "recommendation_intro": "Based on what you told me, here are some recommendations:",
            "follow_up": "Would you like more recommendations or something different?"
        }
    
    async def handle_message(self, user_id: str, message: str) -> Dict:
        """Process user message and generate response"""
        try:
            # Save user message to conversation history
            await save_user_message(user_id, message)
            
            if not message.strip():
                return self._format_response(self.responses["no_input"])
            
            # Get recommendations
            result = await self.recommender.recommend_songs(user_id, message)
            
            # Format response with recommendations
            response = {
                "text": f"{self.responses['recommendation_intro']}\n\n"
                        f"{result['explanation']}\n\n"
                        f"{self.responses['follow_up']}",
                "recommendations": result["recommendations"],
                "analysis": result["analysis"]
            }
            
            # Save bot response to conversation history
            await save_user_message(user_id, response["text"], is_bot=True)
            
            return response
        except Exception as e:
            logger.error(f"Chat handling failed: {e}")
            return self._format_response("Sorry, I couldn't process your request. Please try again.")
    
    async def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Get formatted conversation history"""
        history = await get_user_conversation_history(user_id, limit=10)
        return [{
            "message": msg.message,
            "is_bot": msg.is_bot,
            "timestamp": msg.timestamp.isoformat()
        } for msg in history]
    
    def _format_response(self, text: str) -> Dict:
        """Format simple text response"""
        return {
            "text": text,
            "recommendations": [],
            "analysis": {}
        }