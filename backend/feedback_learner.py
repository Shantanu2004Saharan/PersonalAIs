import json
import os
from collections import defaultdict

class FeedbackLearner:
    def __init__(self, file_path="feedback_store.json"):
        self.file_path = file_path
        self.feedback = defaultdict(dict)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                self.feedback.update(json.load(f))

    import json
import os
from collections import defaultdict

class FeedbackLearner:
    def __init__(self, file_path="feedback_store.json"):
        self.file_path = file_path
        self.feedback = defaultdict(dict)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                self.feedback.update(json.load(f))

    def _save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.feedback, f, indent=2)

    def get_user_feedback(self, user_id):
        return self.feedback.get(user_id, {})

def save_feedback(self, user_id, track_id, liked):
        self.feedback[user_id][track_id] = liked
        self._save()

