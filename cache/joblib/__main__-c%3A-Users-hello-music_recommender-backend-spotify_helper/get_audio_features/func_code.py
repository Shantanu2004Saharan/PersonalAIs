# first line: 34
@memory.cache
def get_audio_features(track_ids):
    """
    Given a list of track IDs, fetch their audio features using Spotify API.
    """
    if not track_ids:
        logging.warning("No track IDs provided for fetching audio features.")
        return []

    # Limit Spotify API to max 100 tracks per request
    max_tracks_per_request = 100
    audio_features = []

    try:
        token = sp.auth_manager.get_access_token()
        token = token['access_token']
        headers = {"Authorization": f"Bearer {token}"}


        for i in range(0, len(track_ids), max_tracks_per_request):
            batch = track_ids[i:i + max_tracks_per_request]
            ids_param = ','.join(batch)

            logging.info(f"Requesting audio features for batch: {ids_param[:100]}...")

            response = requests.get(
                f"https://api.spotify.com/v1/audio-features?ids={ids_param}",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            if data and "audio_features" in data:
                audio_features.extend([af for af in data["audio_features"] if af])
            else:
                logging.warning(f"No audio features returned for batch: {batch}")

        logging.info(f"Successfully fetched audio features for {len(audio_features)} tracks.")
        return audio_features

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching audio features: {e}")
        if e.response is not None:
            logging.error(f"Response: {e.response.text}")
        return []
