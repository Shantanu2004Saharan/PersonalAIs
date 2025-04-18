import streamlit as st
from chatbot import ChatbotController
import unittest
from unittest.mock import MagicMock, patch

# Configuration
st.set_page_config(page_title="🎵 PersonalAIs", layout="centered")

# Initialize controller
controller = ChatbotController()

def music_companion_app():
    """Main application function"""
    st.title("🎧 PersonalAIs: Music Companion")
    st.write("Get personalized music recommendations based on your vibe!")

    query = st.text_input("What type of music do you feel like today?")
    mood = {
        "valence": st.slider("Mood (positivity)", 0.0, 1.0, 0.5),
        "energy": st.slider("Energy Level", 0.0, 1.0, 0.5),
        "danceability": st.slider("Danceability", 0.0, 1.0, 0.5),
    }

    if st.button("Get Recommendations"):
        with st.spinner("Fetching vibes..."):
            results = controller.handle_input(query, mood)
            display_results(results)

def display_results(results):
    """Display results in Streamlit UI"""
    if results:
        st.success("Here's your playlist 🎶")
        for i, r in enumerate(results, 1):
            st.markdown(f"**{i}. {r['name']}** — {', '.join(r['artists'])}")
            if r.get('preview_url'):
                st.audio(r['preview_url'], format='audio/mp3')
            st.markdown(f"[🔗 Open on Spotify]({r['external_url']})")

        if st.button("🎵 Create Playlist"):
            url = controller.create_playlist_from_results("Streamlit Playlist", results)
            st.markdown(f"✅ Playlist created: [Open it here]({url})")
    else:
        st.error("No songs matched your vibe 😢")

class TestMusicCompanion(unittest.TestCase):
    """Unit tests for the music companion app"""
    
    def setUp(self):
        self.controller_mock = MagicMock(spec=ChatbotController)
        
    def test_app_initialization(self):
        """Test that the app initializes correctly"""
        with patch('__main__.controller', self.controller_mock):
            app = music_companion_app()
            self.assertIsNotNone(app)
    
    def test_recommendation_flow(self):
        """Test recommendation flow with mock data"""
        self.controller_mock.handle_input.return_value = [{
            'name': 'Test Song',
            'artists': ['Test Artist'],
            'preview_url': 'https://example.com/preview.mp3',
            'external_url': 'https://example.com/track'
        }] 
        
        with patch('__main__.controller', self.controller_mock):
            test_query = "happy songs"
            test_mood = {"valence": 0.8, "energy": 0.7, "danceability": 0.6}
            
            results = controller.handle_input(test_query, test_mood)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['name'], 'Test Song')

    def test_display_results(self):
        """Test results display functionality"""
        test_results = [{
            'name': 'Test Song',
            'artists': ['Artist1', 'Artist2'],
            'preview_url': 'https://example.com/preview.mp3',
            'external_url': 'https://example.com/track'
        }]
        
        # This would need a Streamlit test context in practice
        display_results(test_results)

# Separate test file would be better, but for demonstration:
if st.secrets.get("RUN_TESTS", False):
    import io
    from contextlib import redirect_stdout
    
    st.subheader("Running Tests...")
    buffer = io.StringIO()
    
    with redirect_stdout(buffer):
        runner = unittest.TextTestRunner(stream=buffer)
        suite = unittest.TestLoader().loadTestsFromTestCase(TestMusicCompanion)
        result = runner.run(suite)
    
    st.text(buffer.getvalue())
    st.success("✅ Tests completed!" if result.wasSuccessful() else "❌ Tests failed")
else:
    music_companion_app()
