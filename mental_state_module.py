# mental_state_module.py

def get_mental_state(emotion_label):
    """Rule-based mapping from emotion → mental state."""
    if emotion_label in ['Happy', 'Surprise']:
        return "Relaxed / Positive"
    elif emotion_label in ['Sad', 'Fear', 'Disgust']:
        return "Stressed / Anxious"
    elif emotion_label == 'Angry':
        return "Possible Stress"
    else:
        return "Neutral / Low Mood"

# Optional: Convert emotion probabilities into a weighted risk score
def estimate_stress_score(predictions, emotion_labels):
    """Compute simple weighted stress level from probabilities."""
    stress_weights = {
        'Happy': 0.1, 'Surprise': 0.1, 'Neutral': 0.4,
        'Sad': 0.8, 'Fear': 0.9, 'Angry': 0.7, 'Disgust': 0.85
    }
    score = sum(predictions[i] * stress_weights[emotion_labels[i]] for i in range(len(emotion_labels)))
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Moderate"
    else:
        return "High"
