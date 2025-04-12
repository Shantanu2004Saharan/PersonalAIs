document.getElementById('recommendationForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const userId = document.getElementById('userId').value;
    const text = document.getElementById('description').value;

    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = 'Loading...';

    try {
    const response = await fetch('http://localhost:8000/recommend', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json'
        },
        body: JSON.stringify({
        user_id: userId,
        text: text
        })
    });

    const data = await response.json();

    if (data.status === 'success') {
        const songs = data.recommendations;
        if (songs.length === 0) {
        resultsDiv.innerHTML = '<p>No recommendations found.</p>';
        } else {
        resultsDiv.innerHTML = '<h3>Top Recommendations:</h3>';
        songs.forEach(song => {
            resultsDiv.innerHTML += `
            <div class="song">
                <strong>${song.title}</strong> by ${song.artist} <br>
                Match Score: ${song.match_score.toFixed(2)}
                <hr>
            </div>
            `;
        });
        }
    } else {
        resultsDiv.innerHTML = '<p>Failed to get recommendations.</p>';
    }
    } catch (err) {
    console.error(err);
    resultsDiv.innerHTML = '<p>Error occurred while fetching recommendations.</p>';
    }
});
