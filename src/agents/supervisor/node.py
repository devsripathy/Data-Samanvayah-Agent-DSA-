"""Supervisor agent node."""
from src.core.state import DSAState
from src.utils.logger import get_logger
from typing_extensions import TypedDict


logger = get_logger(__name__)

async def supervisor_node(state: DSAState) -> dict:
    """Evaluates current state and routes to the next agent."""
    state.update_status("supervising", "supervisor")
    
    # Logic to determine next step based on critic feedback or initial flow
    if state.critic_context and not state.critic_context.passed:
        if state.retry_counter < 3:
            state.retry_counter += 1
            state.metadata["next_agent"] = "planner"
            state.append_log(f"Critic failed. Retrying ({state.retry_counter}/3).")
        else:
            state.metadata["next_agent"] = "end"
            state.add_error("Max retries exceeded.")
    else:
        # Default flow handled by graph edges, but we can override here
        pass 
        
    return {"retry_counter": state.retry_counter, "metadata": state.metadata}
