"""Critic agent node for evaluation."""
from src.core.state import DSAState, CriticContext
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def critic_node(state: DSAState) -> dict:
    """
    Evaluates training results and determines if they pass quality thresholds.
    
    Includes confidence scoring for human-in-the-loop workflows:
    - High confidence (>0.8): Deterministic routing
    - Medium confidence (0.6-0.8): Potential HITL intervention
    - Low confidence (<0.6): Strong HITL candidate for ambiguous cases
    """
    state.update_status("critiquing", "critic")
    state.append_log("Evaluating model performance.")
    
    # Get metrics from training
    metrics = state.training_results.metrics if state.training_results else {}
    accuracy = metrics.get("accuracy", 0)
    f1 = metrics.get("f1", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    
    # Determine pass/fail based on accuracy threshold
    passed = accuracy > 0.90
    
    # Calculate confidence score (simple average of key metrics)
    # If metrics are missing, confidence is lower
    available_metrics = sum([
        1 for m in [accuracy, f1, precision, recall] if m > 0
    ])
    
    if available_metrics > 0:
        confidence = (accuracy + f1 + precision + recall) / (4 if available_metrics == 4 else available_metrics)
    else:
        confidence = 0.0
    
    # Determine issues
    issues = []
    if accuracy <= 0.90:
        issues.append(f"Accuracy below 90% threshold ({accuracy:.2%})")
    if f1 > 0 and f1 < 0.85:
        issues.append(f"F1 score below optimal ({f1:.2%})")
    if len(metrics) == 0:
        issues.append("No metrics available for evaluation")
    
    # Build critic context
    context = CriticContext(
        passed=passed,
        confidence=confidence,
        issues_found=issues if not passed else [],
        retry_reason="Improve feature engineering or hyperparameters" if not passed else None,
        recommendations=[
            "Try hyperparameter tuning",
            "Increase training data",
            "Consider ensemble methods",
        ] if not passed else []
    )
    
    state.append_log(
        f"Critique complete. Passed: {passed}, Confidence: {confidence:.2f}"
    )
    
    return {"critic_context": context}
