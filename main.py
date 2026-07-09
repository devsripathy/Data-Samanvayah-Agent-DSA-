"""Main execution script for DSA."""
import asyncio
from src.core.graph import dsa_graph
from src.core.state import DSAState
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def run_pipeline(dataset_path: str):
    """Initializes state and runs the DSA pipeline."""
    initial_state = DSAState(
        dataset=dataset_path,
        status="starting"
    )
    initial_state.append_log("Pipeline initialized.")
    
    logger.info(f"Starting DSA execution for {dataset_path}")
    final_state = await dsa_graph.ainvoke(initial_state)
    
    logger.info(f"Pipeline completed with status: {final_state.status}")
    return final_state

if __name__ == "__main__":
    asyncio.run(run_pipeline("data/sample.csv"))
