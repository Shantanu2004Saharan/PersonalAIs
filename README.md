
# PersonalAIs - AI-Powered Music Recommender

Welcome to **PersonalAIs** — your intelligent music companion.  
Simply tell the app your mood, activity, or vibe, and it will recommend **personalized playlists** curated just for you.  
You can also **create playlists directly on your Spotify account**.

---

## Table of Contents
- [About](#about)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Running the App](#running-the-app)
- [How Spotify Login Works](#how-spotify-login-works)
- [API Overview](#api-overview)
- [Frontend Walkthrough](#frontend-walkthrough)
- [Features](#features)
- [Credits](#credits)

---

## About

**PersonalAIs** is a full-stack music recommendation system that:
- Understands your **mood** and **activity** using **natural language processing (NLP)**.
- Searches and ranks songs from **Spotify** based on emotional and semantic similarity.
- Remembers your **conversation history** and **preferences**.
- Allows you to **create custom playlists** directly on your **Spotify account**.
- Continuously improves recommendations based on your **feedback**.

---

## Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy, AsyncIO
- **ML/NLP Models**: 
  - Sentence Transformers (`all-MiniLM-L6-v2`)
  - Huggingface Emotion Detection (`distilbert-go-emotions`)
  - FAISS (vector similarity search)
- **Music Data**: Spotify Web API (`spotipy`)
- **Frontend**: 
  - Custom HTML/CSS Frontend
  - Optional: Streamlit Application
- **Database**: Async SQLite (`aiosqlite`)
- **Authentication**: Spotify OAuth 2.0

---

## Setup and Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/PersonalAIs.git
   cd PersonalAIs
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Spotify App Credentials**  
   This project already has a working Spotify App setup:
   ```
   client_id = "736bb144677e448dad56d2fe2ab70cd0"
   client_secret = "d7beffe6e8d740deb7e1ddd9a111c88f"
   redirect_uri = "http://127.0.0.1:8000/callback"
   ```
   You can directly log in and start using your own Spotify account.

---

## Running the App

1. **Start the backend server**
   ```bash
   uvicorn main:app --reload
   ```

2. **Open the Frontend**
   - Open `frontend/index.html` in your browser.
   - Alternatively, serve the frontend using a simple static server (e.g., `python -m http.server`).

3. **(Optional) Run the Streamlit Application**
   ```bash
   streamlit run Streamlit_app.py
   ```

---

## How Spotify Login Works

- Click the **Login with Spotify** button on the web page.
- Authenticate using your personal Spotify account.
- No additional setup is required — the app is pre-linked with Spotify.
- After logging in, you can:
  - Receive personalized music recommendations.
  - Create and save playlists directly from the chatbot.

> *You will always use your **own Spotify account** — not the developer's.*

---

## API Overview

| Endpoint              | Method | Purpose                                    |
|:----------------------|:-------|:-------------------------------------------|
| `/api/chat`            | POST   | Chat and receive song recommendations      |
| `/api/create_playlist` | POST   | Create a playlist from recommended songs   |
| `/api/feedback`        | POST   | Submit feedback for improved suggestions   |
| `/login`               | GET    | Start Spotify OAuth login process          |
| `/callback`            | GET    | Spotify OAuth callback endpoint            |

---

## Frontend Walkthrough

- Users provide a **User ID** and a **Mood/Activity**.
- Conversations between the user and the bot appear dynamically.
- Recommended tracks display the following information:
  - Track Name
  - Artist
  - Audio preview
  - Spotify link
- Users can name their playlists and create them directly from the interface.

---

## Features

- Emotion and Activity-Based Song Recommendations
- Conversational Memory for Personalized Experience
- Direct Spotify Playlist Creation
- Adaptive Learning from User Feedback
- Asynchronous and Fast Backend
- Custom-Built, User-Friendly Frontend
- Quick Setup — No Environment Variables Required

---

## Credits

- **Developer**: Shantanu (IIT Bombay)
- **Technologies Used**: Spotify API, Huggingface Transformers, Sentence-Transformers, FAISS, FastAPI

---

# Thank You for Using PersonalAIs
> *"Music is the shorthand of emotion." – Leo Tolstoy*
