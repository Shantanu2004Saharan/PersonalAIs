import streamlit as st
from chatbot import ChatbotController

st.set_page_config(page_title="🎵 PersonalAIs", layout="centered")

controller = ChatbotController()

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
        if results:
            st.success("Here's your playlist 🎶")
            for i, r in enumerate(results, 1):
                st.markdown(f"**{i}. {r['name']}** — {', '.join(r['artists'])}")
                st.audio(r['preview_url'] or "", format='audio/mp3')
                st.markdown(f"[🔗 Open on Spotify]({r['external_url']})")

            if st.button("🎵 Create Playlist"):
                url = controller.create_playlist_from_results("Streamlit Playlist", results)
                st.markdown(f"✅ Playlist created: [Open it here]({url})")
        else:
            st.error("No songs matched your vibe 😢")


