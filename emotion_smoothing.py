from collections import Counter, deque


class EmotionSmoother:
    def __init__(self, window_size=10, min_votes=6, default_emotion="neutral"):
        self.window_size = window_size
        self.min_votes = min_votes
        self.default_emotion = default_emotion
        self.buffer = deque(maxlen=window_size)
        self.current_emotion = default_emotion

    def update(self, emotion):
        self.buffer.append(emotion)

        if len(self.buffer) < self.window_size:
            return self.current_emotion

        most_common_emotion, votes = Counter(self.buffer).most_common(1)[0]

        if votes >= self.min_votes:
            self.current_emotion = most_common_emotion

        return self.current_emotion

    def reset(self):
        self.buffer.clear()
        self.current_emotion = self.default_emotion