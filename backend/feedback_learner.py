import json
import os
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

DEFAULT_FILE_PATH = "feedback_store.json"

# Separate function to match usage in main.py
async def save_feedback(user_id: str, track_id: str, liked: bool):
    """
    Async function to save feedback for a user to a JSON file.
    """
    try:
        if not os.path.exists(DEFAULT_FILE_PATH):
            with open(DEFAULT_FILE_PATH, "w") as f:
                json.dump({}, f)

        with open(DEFAULT_FILE_PATH, "r") as f:
            feedback = json.load(f)

        user_feedback = feedback.get(user_id, {})
        user_feedback[track_id] = liked
        feedback[user_id] = user_feedback

        with open(DEFAULT_FILE_PATH, "w") as f:
            json.dump(feedback, f, indent=2)

        logger.info(f"Feedback saved: user={user_id}, track={track_id}, liked={liked}")
    except Exception as e:
        logger.error(f"Error saving feedback to JSON: {e}", exc_info=True)
        raise


class FeedbackLearner:
    def __init__(self, file_path: str = DEFAULT_FILE_PATH):
        self.file_path = file_path
        self.feedback = defaultdict(dict)

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    self.feedback.update(json.load(f))
                except json.JSONDecodeError:
                    logger.warning("Feedback JSON file was empty or invalid. Starting fresh.")
                    self.feedback = defaultdict(dict)

    def _save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.feedback, f, indent=2)

    def get_user_feedback(self, user_id):
        return self.feedback.get(user_id, {})

    async def process_feedback(self, user_id: str, track_id: str, liked: bool):
        """
        Simulate real-time learning logic after saving feedback.
        """
        try:
            self.feedback[user_id][track_id] = liked
            self._save()
            logger.info(f"Processed feedback for user={user_id}, track={track_id}, liked={liked}")
        except Exception as e:
            logger.error(f"Error processing feedback: {e}", exc_info=True)
            raise

    def run_tests(self):
        print("Running test 1...")
        self.feedback.clear()
        self.save_feedback_sync("user1", "track123", True)
        feedback = self.get_user_feedback("user1")
        assert feedback["track123"] is True
        print("Test 1 passed.")

        print("Running test 2...")
        self.save_feedback_sync("user1", "track456", False)
        feedback = self.get_user_feedback("user1")
        assert feedback["track456"] is False
        print("Test 2 passed.")

        print("Running test 3...")
        with open(self.file_path, "r") as f:
            data = json.load(f)
            assert data["user1"]["track123"] is True
            assert data["user1"]["track456"] is False
        print("Test 3 passed.")

        os.remove(self.file_path)
        print("All tests passed!")

    def save_feedback_sync(self, user_id: str, track_id: str, liked: bool):
        self.feedback[user_id][track_id] = liked
        self._save()


# Example usage
if __name__ == "__main__":
    learner = FeedbackLearner(file_path="test_feedback_store.json")
    learner.run_tests()
