"""
Emotion detection module for AgentX chatbot.
Uses a lightweight approach for detecting user emotions from text.
"""

import re
from typing import Dict, List, Optional
import requests
import json


class EmotionDetector:
    """Lightweight emotion detection using keyword-based approach with optional API fallback."""
    
    def __init__(self):
        # Emotion keywords and patterns
        self.emotion_patterns = {
            'happy': [
                r'\b(?:happy|joy|excited|great|awesome|fantastic|wonderful|amazing|love|like|enjoy|pleased|delighted)\b',
                r'\b(?:😊|😄|😃|😁|😆|😍|🥰|😘|🤗|👍|👏|🎉|🎊)\b'
            ],
            'sad': [
                r'\b(?:sad|depressed|down|upset|disappointed|hurt|broken|crying|tears|grief|mourning)\b',
                r'\b(?:😢|😭|😔|😞|😟|😿|💔|😿)\b'
            ],
            'angry': [
                r'\b(?:angry|mad|furious|rage|annoyed|irritated|frustrated|pissed|hate|disgusted)\b',
                r'\b(?:😠|😡|🤬|😤|💢|🔥|👿|😾)\b'
            ],
            'anxious': [
                r'\b(?:anxious|worried|nervous|stressed|panic|fear|scared|afraid|concerned|troubled)\b',
                r'\b(?:😰|😨|😱|😳|😟|😕|🤔|😓)\b'
            ],
            'confused': [
                r'\b(?:confused|lost|unclear|don\'t understand|help|stuck|problem|issue|trouble)\b',
                r'\b(?:😕|🤔|😵|😵‍💫|🤷|❓|❔)\b'
            ],
            'grateful': [
                r'\b(?:thank|thanks|grateful|appreciate|blessed|fortunate|lucky|gratitude)\b',
                r'\b(?:🙏|😇|😊|💝|🎁)\b'
            ]
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            'very': 2.0,
            'really': 1.8,
            'extremely': 2.5,
            'super': 1.5,
            'so': 1.3,
            'quite': 1.2,
            'slightly': 0.5,
            'a bit': 0.7,
            'kind of': 0.6
        }
    
    def detect_emotion(self, text: str) -> Dict[str, any]:
        """
        Detect emotion from text input.
        
        Returns:
            Dict with 'emotion', 'confidence', 'intensity', and 'keywords'
        """
        if not text or len(text.strip()) < 3:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'intensity': 'low',
                'keywords': []
            }
        
        text_lower = text.lower()
        emotion_scores = {}
        found_keywords = {}
        
        # Score each emotion
        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0
            keywords = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    score += len(matches)
                    keywords.extend(matches)
            
            # Check for intensity modifiers
            for modifier, multiplier in self.intensity_modifiers.items():
                if modifier in text_lower:
                    # Look for emotion words near intensity modifiers
                    for pattern in patterns:
                        if re.search(f'{modifier}.*{pattern}|{pattern}.*{modifier}', text_lower):
                            score *= multiplier
                            break
            
            if score > 0:
                emotion_scores[emotion] = score
                found_keywords[emotion] = keywords
        
        # Determine primary emotion
        if not emotion_scores:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'intensity': 'low',
                'keywords': []
            }
        
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[primary_emotion]
        total_score = sum(emotion_scores.values())
        confidence = min(max_score / max(total_score, 1), 1.0)
        
        # Determine intensity
        if confidence > 0.8:
            intensity = 'high'
        elif confidence > 0.5:
            intensity = 'medium'
        else:
            intensity = 'low'
        
        return {
            'emotion': primary_emotion,
            'confidence': round(confidence, 2),
            'intensity': intensity,
            'keywords': found_keywords.get(primary_emotion, [])
        }
    
    def get_emotion_prompt_addition(self, emotion_data: Dict) -> str:
        """Generate system prompt addition based on detected emotion."""
        emotion = emotion_data['emotion']
        intensity = emotion_data['intensity']
        confidence = emotion_data['confidence']
        
        if emotion == 'neutral' or confidence < 0.3:
            return ""
        
        emotion_responses = {
            'happy': f"User appears {emotion} ({intensity} confidence). Respond with enthusiasm and positive energy.",
            'sad': f"User seems {emotion} ({intensity} confidence). Respond with empathy, comfort, and understanding.",
            'angry': f"User appears {emotion} ({intensity} confidence). Respond calmly, acknowledge their feelings, and offer solutions.",
            'anxious': f"User seems {emotion} ({intensity} confidence). Respond with reassurance, patience, and clear guidance.",
            'confused': f"User appears {emotion} ({intensity} confidence). Provide clear, step-by-step explanations and ask clarifying questions.",
            'grateful': f"User seems {emotion} ({intensity} confidence). Acknowledge their gratitude warmly and continue being helpful."
        }
        
        return emotion_responses.get(emotion, "")


# Global instance
emotion_detector = EmotionDetector()


def detect_user_emotion(text: str) -> Dict[str, any]:
    """Convenience function to detect emotion from user text."""
    return emotion_detector.detect_emotion(text)


def get_emotion_context_for_prompt(text: str) -> str:
    """Get emotion context to add to system prompt."""
    emotion_data = detect_user_emotion(text)
    return emotion_detector.get_emotion_prompt_addition(emotion_data)
