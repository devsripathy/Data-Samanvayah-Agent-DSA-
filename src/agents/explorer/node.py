"""Explorer agent node for EDA and preprocessing."""
from src.core.state import DSAState
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def explorer_node(state: DSAState) -> dict:
    """Performs Exploratory Data Analysis and applies preprocessing."""
    state.update_status("exploring", "explorer")
    state.append_log("Executing EDA and preprocessing steps.")
    
    df = state.cleaned_dataset.copy()
    
    # Apply preprocessing steps from planner
    for step in state.planner_context.preprocessing_steps:
        if step == "impute_missing":
            df = df.fillna(df.mean(numeric_only=True))
        state.append_log(f"Applied preprocessing: {step}")
        
    state.artifacts["eda_report"] = {"shape": df.shape, "dtypes": df.dtypes.astype(str).to_dict()}
    return {"cleaned_dataset": df}
