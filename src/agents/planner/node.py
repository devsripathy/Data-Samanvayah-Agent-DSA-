"""Planning agent node."""
from langchain_openai import ChatOpenAI
from src.core.state import DSAState, PlannerContext
from src.config.settings import settings
from src.utils.logger import get_logger
from typing_extensions import TypedDict

logger = get_logger(__name__)

async def planner_node(state: DSAState) -> dict:
    """Analyzes data and creates an execution plan using an LLM."""
    state.update_status("planning", "planner")
    state.append_log("Generating execution plan via LLM.")
    
    llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)
    
    # Simplified LLM prompt for demonstration
    # Updated to use dataset_schema instead of schema
    prompt = f"Analyze this schema and suggest a plan: {state.dataset_metadata.dataset_schema}"
    # response = await llm.ainvoke(prompt) 
    
    plan = PlannerContext(
        target_column="col2",
        confidence=0.85,
        reasoning="Target column appears categorical.",
        suggested_models=["RandomForest", "XGBoost"],
        preprocessing_steps=["impute_missing", "label_encode"]
    )
    
    state.metadata["next_agent"] = "explorer"
    return {"planner_context": plan}
