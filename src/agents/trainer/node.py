"""Trainer agent node for model training."""
from src.core.state import DSAState, TrainingResults
from src.utils.logger import get_logger
from typing_extensions import TypedDict
import time

logger = get_logger(__name__)

async def trainer_node(state: DSAState) -> dict:
    """Trains models based on the planner's suggestions."""
    state.update_status("training", "trainer")
    state.append_log("Starting model training loop.")
    
    start_time = time.time()
    
    # Simulate training
    best_model_name = state.planner_context.suggested_models[0]
    metrics = {"accuracy": 0.92, "f1_score": 0.89}
    
    training_time = time.time() - start_time
    
    results = TrainingResults(
        best_model=best_model_name,
        task_type="classification",
        metrics=metrics,
        leaderboard=[{"model": best_model_name, "score": 0.92}],
        feature_importance={"col1": 0.7, "col2": 0.3},
        training_time=training_time,
        artifacts={"model_path": f"artifacts/{best_model_name}.pkl"}
    )
    
    state.append_log(f"Training complete. Best model: {best_model_name}")
    return {"training_results": results}
