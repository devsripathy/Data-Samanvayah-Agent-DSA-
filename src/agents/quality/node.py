"""Data quality assessment agent node."""
import pandas as pd
from src.core.state import DSAState, DatasetMetadata
from src.utils.logger import get_logger
from typing_extensions import TypedDict


logger = get_logger(__name__)

async def quality_node(state: DSAState) -> dict:
    """Performs initial data profiling and quality checks."""
    state.update_status("checking_quality", "quality")
    state.append_log("Running initial data quality checks.")
    
    # Simulate loading data
    df = pd.DataFrame({"col1": [1, 2, None], "col2": ["a", "b", "c"]})
    
    metadata = DatasetMetadata(
        dataset_path=state.dataset,
        dataset_name="sample_data",
        dataset_size=df.memory_usage().sum(),
        rows=len(df),
        columns=len(df.columns),
        file_type="csv",
        dataset_schema={col: str(dtype) for col, dtype in df.dtypes.items()}, # Updated to dataset_schema
        missing_summary=df.isnull().sum().to_dict()
    )
    
    state.append_log(f"Data profiled: {metadata.rows} rows, {metadata.columns} cols.")
    return {"cleaned_dataset": df, "dataset_metadata": metadata}
