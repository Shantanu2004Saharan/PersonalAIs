# first line: 20
@memory.cache
def get_tracks_by_query(query):
    """
    Fetch tracks based on a query string.
    """
    try:
        results = sp.search(q=query, limit=50, type='track')
        tracks = results['tracks']['items']
        logging.info(f"Fetched {len(tracks)} tracks for query: {query}")
        return tracks
    except Exception as e:
        logging.error(f"Error fetching tracks for query '{query}': {e}")
        return []
