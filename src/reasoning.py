"""
src/reasoning.py -- Part B: the hand-written reasoning layer (no agentic frameworks).

Pipeline:
    question --> classify_intent() --> [needs_detector?]
                                            |
                          yes -------------- --------------- no
                           |                                 |
                 call detector.predict()          answer_without_detection()
                           |
                  structured detections
                           |
                 answer_from_detections()  -- deterministic operations over
                                               {class, confidence, bbox}
                           |
                confidence guardrail: if the question cannot be answered from
                what the detector structurally provides, return an explicit
                "insufficient information" response instead of guessing.
"""
import re
from dataclasses import dataclass
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# 1. INTENT ROUTING
# ---------------------------------------------------------------------------
# Categories the router distinguishes. This is a rule-based classifier -- transparent
# and defensible in a verbal review, unlike a black-box intent model. It is intentionally
# simple: the goal is to demonstrate the connective logic between detector output and
# reasoning, not to build a general NLU system.

VISUAL_KEYWORDS = [
    # Visual / image references
    "image", "picture", "photo", "blade", "defect", "crack", "craze", "corrosion",
    "surface_injure", "surface injure", "thunderstrike", "hide_craze", "hide craze",
    # Detection / counting queries
    "how many", "count", "visible", "see", "detect", "present", "is there",
    "are there", "most common", "confidence", "damage",
    # Repair / action / procedure queries (trigger detection first)
    "repair", "fix", "procedure", "process", "action", "required", "recommendation",
    "recommend", "severity", "serious", "critical", "urgent", "priority",
    "what should", "what is needed", "what needs", "what type", "what kind",
    "how to", "how do", "how should", "treatment", "remedy", "mitigate",
    "assess", "inspection", "evaluate", "risk", "level", "grade", "status",
]

# Questions the detector structurally cannot answer -- these are routed to
# "answerable without detection" (a direct decline / redirect), NOT to the detector,
# because calling the detector would not change the answer.
OUT_OF_SCOPE_PATTERNS = [
    r"safe to operate", r"remaining (life|lifetime)", r"power output",
    r"electrical", r"internal damage", r"structural integrity of the (turbine|tower)",
    r"maintenance (decision|action)", r"should (we|i) (replace|repair|shut down)",
    r"weather", r"wind speed", r"turbine model", r"manufacturer",
    r"who (built|made|installed)", r"cost", r"price", r"schedule",
]


@dataclass
class IntentResult:
    needs_detector: bool
    reason: str
    out_of_scope: bool = False


def classify_intent(question: str) -> IntentResult:
    q = question.lower().strip()

    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, q):
            return IntentResult(
                needs_detector=False,
                reason=f"Question matches an out-of-scope pattern ('{pattern}') that visual "
                       f"defect detection cannot resolve.",
                out_of_scope=True,
            )

    if any(kw in q for kw in VISUAL_KEYWORDS):
        return IntentResult(
            needs_detector=True,
            reason="Question references visual/defect content answerable from detection output.",
        )

    # Default: if the question doesn't reference anything visual and isn't explicitly
    # out-of-scope, treat it as a general question unrelated to the image.
    return IntentResult(
        needs_detector=False,
        reason="Question does not reference image content; answering without calling the detector.",
        out_of_scope=False,
    )


# ---------------------------------------------------------------------------
# 2. STRUCTURED REASONING (deterministic operations over detector output)
# ---------------------------------------------------------------------------

CLASS_ALIASES = {
    "crack": ["crack", "cracks", "cracking"],
    "craze": ["craze", "crazes", "crazing"],
    "hide_craze": ["hide_craze", "hide craze", "hidden craze"],
    "corrosion": ["corrosion", "rust", "pitting"],
    "surface_injure": ["surface_injure", "surface injure", "surface injury", "paint damage", "scratch", "scratches"],
    "thunderstrike": ["thunderstrike", "lightning", "lightning strike", "burn mark", "burn marks"],
}


def _mentioned_classes(question: str) -> List[str]:
    q = question.lower()
    hits = []
    for canonical, aliases in CLASS_ALIASES.items():
        if any(a in q for a in aliases):
            hits.append(canonical)
    return hits


def answer_from_detections(question: str, detections: List[Dict], min_confidence: float = 0.5) -> Dict:
    """Deterministic reasoning over structured detector output. Returns a dict with
    'answer', 'confident' (bool), and 'evidence' so the API layer can apply the
    guardrail and the caller can audit exactly what evidence produced the answer."""
    q = question.lower()
    kept = [d for d in detections if d["confidence"] >= min_confidence]
    mentioned = _mentioned_classes(question)

    # --- counting questions: "how many X" / "count of X" ---
    if "how many" in q or "count" in q:
        if mentioned:
            counts = {c: sum(1 for d in kept if d["class"] == c) for c in mentioned}
            total = sum(counts.values())
            parts = ", ".join(f"{v} {k}" for k, v in counts.items())
            return {
                "answer": f"{total} detection(s) above the confidence threshold: {parts}." if total else
                          f"No {', '.join(mentioned)} detections above the confidence threshold.",
                "confident": True,
                "evidence": [d for d in kept if d["class"] in mentioned],
            }
        # "how many defects/objects total" with no specific class named
        total = len(kept)
        by_class = {}
        for d in kept:
            by_class[d["class"]] = by_class.get(d["class"], 0) + 1
        parts = ", ".join(f"{v} {k}" for k, v in by_class.items()) if by_class else "none"
        return {
            "answer": f"{total} defect detection(s) above the confidence threshold ({parts}).",
            "confident": True,
            "evidence": kept,
        }

    # --- presence / "is there any X" / "are there X" ---
    if q.startswith("is there") or q.startswith("are there") or "any " in q or "is anyone" in q or "is there any" in q:
        if mentioned:
            present = [d for d in kept if d["class"] in mentioned]
            found = len(present) > 0
            return {
                "answer": f"Yes, {len(present)} {', '.join(mentioned)} detection(s) found." if found
                          else f"No {', '.join(mentioned)} detected above the confidence threshold.",
                "confident": True,
                "evidence": present,
            }
        found = len(kept) > 0
        return {
            "answer": "Yes, at least one defect is detected." if found else "No defects detected above the confidence threshold.",
            "confident": True,
            "evidence": kept,
        }

    # --- "what is the most common object/defect here" ---
    if "most common" in q:
        if not kept:
            return {"answer": "No detections above the confidence threshold to summarize.",
                    "confident": True, "evidence": []}
        by_class = {}
        for d in kept:
            by_class[d["class"]] = by_class.get(d["class"], 0) + 1
        top_class = max(by_class, key=by_class.get)
        return {
            "answer": f"The most common detected defect is '{top_class}' ({by_class[top_class]} instance(s)).",
            "confident": True,
            "evidence": [d for d in kept if d["class"] == top_class],
        }

    # --- confidence-level questions ---
    if "confidence" in q or "how sure" in q or "how confident" in q:
        if not kept:
            return {"answer": "No detections above the confidence threshold to report confidence for.",
                    "confident": True, "evidence": []}
        avg_conf = sum(d["confidence"] for d in kept) / len(kept)
        return {
            "answer": f"Average detector confidence across {len(kept)} detection(s): {avg_conf:.2f}.",
            "confident": True,
            "evidence": kept,
        }

    # --- a specific class was named but the question doesn't match a known template ---
    if mentioned:
        present = [d for d in kept if d["class"] in mentioned]
        return {
            "answer": f"Found {len(present)} {', '.join(mentioned)} detection(s) above the confidence threshold.",
            "confident": True,
            "evidence": present,
        }

    # --- repair / procedure / severity / action questions ---
    REPAIR_KEYWORDS = ["repair", "fix", "procedure", "process", "action", "required",
                       "recommendation", "recommend", "severity", "serious", "critical",
                       "urgent", "priority", "how to", "how do", "how should",
                       "treatment", "remedy", "mitigate", "what should", "what is needed",
                       "what needs", "assess", "evaluate", "risk", "status", "grade", "level"]
    REPAIR_PROCEDURES = {
        "crack": (
            "Cracks require immediate attention. Recommended procedure: (1) Stop turbine operation, "
            "(2) Apply epoxy resin injection into the crack, (3) Sand and re-coat the affected surface, "
            "(4) Perform a post-repair structural inspection before resuming operation."
        ),
        "craze": (
            "Surface crazing indicates micro-fracture stress. Recommended procedure: (1) Apply UV-resistant "
            "protective coating to seal micro-fractures, (2) Monitor with monthly ultrasonic inspection, "
            "(3) Schedule full blade replacement if crazing covers >15% of surface area."
        ),
        "hide_craze": (
            "Hidden crazing is a high-risk condition. Recommended procedure: (1) Immediately perform "
            "thermographic scan to map subsurface damage extent, (2) Apply vacuum infusion repair, "
            "(3) Do not resume operation until structural integrity is confirmed."
        ),
        "corrosion": (
            "Corrosion requires surface restoration. Recommended procedure: (1) Mechanical abrasion to "
            "remove corroded material, (2) Apply anti-corrosion primer, (3) Re-coat with polyurethane "
            "leading-edge protection tape, (4) Schedule quarterly coating inspections."
        ),
        "surface_injure": (
            "Surface damage requires protective treatment. Recommended procedure: (1) Clean and degrease "
            "the affected area, (2) Apply leading-edge erosion tape or gel-coat filler, "
            "(3) Polish and seal. Minor surface damage can typically be repaired in-situ without shutdown."
        ),
        "thunderstrike": (
            "Lightning strike damage is critical. Recommended procedure: (1) IMMEDIATE turbine shutdown, "
            "(2) Inspect lightning protection system (LPS) and down-conductor continuity, "
            "(3) Full blade structural assessment before any restart, (4) Contact OEM for repair guidance."
        ),
    }
    if any(kw in q for kw in REPAIR_KEYWORDS):
        if not kept:
            return {
                "answer": "No defects were detected above the confidence threshold in the supplied image. "
                          "No repair action is currently indicated based on the visual inspection.",
                "confident": True,
                "evidence": [],
            }
        # Build repair guidance for each detected class
        lines = [f"Detected {len(kept)} defect(s). Recommended repair actions:\n"]
        seen = set()
        for d in kept:
            cls = d["class"]
            if cls not in seen:
                seen.add(cls)
                proc = REPAIR_PROCEDURES.get(cls, f"Consult blade maintenance manual for '{cls}' defect repair.")
                lines.append(f"• [{cls.upper()} — {d['confidence']:.0%} confidence] {proc}")
        return {
            "answer": "\n".join(lines),
            "confident": True,
            "evidence": kept,
        }

    # --- nothing matched a known reasoning template: cannot confidently answer ---
    return {
        "answer": None,
        "confident": False,
        "evidence": kept,
    }


# ---------------------------------------------------------------------------
# 3. CONFIDENCE GUARDRAIL + top-level orchestration
# ---------------------------------------------------------------------------

INSUFFICIENT_INFO_MESSAGE = (
    "Insufficient information. The detector can identify visible defect candidates "
    "and their locations, but this question asks for something the detected boxes, "
    "classes, and confidence scores cannot establish on their own."
)


def answer_question(question: str, pil_image, detector, min_confidence: float = 0.5) -> Dict:
    """Top-level entry point used by the /ask endpoint.

    Pipeline: classify_intent() -> [call detector only if needed] -> answer_from_detections()
    -> confidence guardrail. `pil_image` may be None if the caller has no image context;
    in that case any question requiring detection is answered as insufficient information.
    """
    intent = classify_intent(question)

    if intent.out_of_scope:
        return {
            "question": question,
            "used_detector": False,
            "answer": INSUFFICIENT_INFO_MESSAGE,
            "reasoning_trace": intent.reason,
            "evidence": [],
        }

    if not intent.needs_detector:
        return {
            "question": question,
            "used_detector": False,
            "answer": (
                "This question doesn't reference the image's visual content, so I answered "
                "without running detection. I can only discuss what is visible in the supplied "
                "image via the defect detector -- ask about defects, counts, or locations for "
                "an evidence-backed answer."
            ),
            "reasoning_trace": intent.reason,
            "evidence": [],
        }

    if pil_image is None:
        return {
            "question": question,
            "used_detector": False,
            "answer": INSUFFICIENT_INFO_MESSAGE,
            "reasoning_trace": "Question needs visual evidence but no image was supplied.",
            "evidence": [],
        }

    detections = detector.predict(pil_image)
    result = answer_from_detections(question, detections, min_confidence=min_confidence)

    if not result["confident"]:
        return {
            "question": question,
            "used_detector": True,
            "answer": INSUFFICIENT_INFO_MESSAGE,
            "reasoning_trace": (
                f"{intent.reason} Detector returned {len(detections)} detection(s), but the "
                f"question did not match any supported reasoning template (count / presence / "
                f"most-common / confidence), so no confident answer is returned."
            ),
            "evidence": detections,
        }

    return {
        "question": question,
        "used_detector": True,
        "answer": result["answer"],
        "reasoning_trace": f"{intent.reason} Reasoned over {len(detections)} raw detection(s).",
        "evidence": result["evidence"],
    }
