"""Critic agent node for evaluation."""
from src.core.state import DSAState, CriticContext
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def critic_node(state: DSAState) -> dict:
    """Evaluates training results and determines if they pass quality thresholds."""
    state.update_status("critiquing", "critic")
    state.append_log("Evaluating model performance.")
    
    metrics = state.training_results.metrics
    passed = metrics.get("accuracy", 0) > 0.90
    
    context = CriticContext(
        passed=passed,
        issues_found=[] if passed else ["Accuracy below 90% threshold"],
        retry_reason="Improve feature engineering" if not passed else None,
        recommendations=["Try hyperparameter tuning"] if not passed else []
    )
    
    state.append_log(f"Critique complete. Passed: {passed}")
    return {"critic_context": context}
