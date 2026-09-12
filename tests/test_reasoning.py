"""
tests/test_reasoning.py -- unit tests for the Part B reasoning layer.
Run with: pytest tests/test_reasoning.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reasoning import classify_intent, answer_from_detections, answer_question, INSUFFICIENT_INFO_MESSAGE


SAMPLE_DETECTIONS = [
    {"class": "crack", "confidence": 0.91, "bbox": [421, 238, 518, 341]},
    {"class": "crack", "confidence": 0.84, "bbox": [50, 60, 120, 140]},
    {"class": "corrosion", "confidence": 0.89, "bbox": [610, 390, 721, 465]},
]


class FakeDetector:
    """Stands in for the real RT-DETR model so reasoning tests don\'t need a GPU or checkpoint."""
    def __init__(self, detections):
        self.detections = detections

    def predict(self, image):
        return self.detections


def test_intent_routes_visual_question_to_detector():
    result = classify_intent("How many cracks are visible?")
    assert result.needs_detector is True


def test_intent_routes_unrelated_question_away_from_detector():
    result = classify_intent("What is the weather like today?")
    assert result.needs_detector is False


def test_intent_routes_out_of_scope_safety_question():
    result = classify_intent("Is this turbine safe to operate?")
    assert result.out_of_scope is True
    assert result.needs_detector is False


def test_counting_question_answered_correctly():
    result = answer_from_detections("How many cracks are visible?", SAMPLE_DETECTIONS)
    assert result["confident"] is True
    assert "2 crack" in result["answer"]


def test_presence_question_answered_correctly():
    result = answer_from_detections("Is there any corrosion?", SAMPLE_DETECTIONS)
    assert result["confident"] is True
    assert "Yes" in result["answer"]


def test_most_common_question():
    result = answer_from_detections("What is the most common object here?", SAMPLE_DETECTIONS)
    assert "crack" in result["answer"]


def test_end_to_end_insufficient_information_example():
    """The specific required example: a question the detector\'s structured output
    cannot answer must return the explicit insufficient-information message, not a guess."""
    detector = FakeDetector(SAMPLE_DETECTIONS)
    result = answer_question("Is this turbine safe to operate?", pil_image=object(), detector=detector)
    assert result["used_detector"] is False
    assert result["answer"] == INSUFFICIENT_INFO_MESSAGE


def test_end_to_end_answerable_example():
    detector = FakeDetector(SAMPLE_DETECTIONS)
    result = answer_question("How many cracks are visible?", pil_image=object(), detector=detector)
    assert result["used_detector"] is True
    assert "2 crack" in result["answer"]


def test_unmatched_visual_question_triggers_guardrail_not_a_guess():
    """A visual question that doesn\'t match any supported reasoning template should also
    fall back to insufficient information rather than fabricating an answer."""
    detector = FakeDetector(SAMPLE_DETECTIONS)
    result = answer_question("Describe the emotional tone of this blade.", pil_image=object(), detector=detector)
    assert result["answer"] == INSUFFICIENT_INFO_MESSAGE
