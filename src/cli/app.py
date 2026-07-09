"""CLI application for DSA."""
import typer
import asyncio
from src.core.graph import dsa_graph
from src.core.state import DSAState

app = typer.Typer(help="Data Samanvayah Agent (DSA) CLI")

@app.command()
def run(
    dataset: str = typer.Argument(..., help="Path to the dataset file"),
    target: str = typer.Option(None, help="Target column for prediction")
):
    """Runs the DSA pipeline on a given dataset."""
    typer.echo(f"Initializing DSA for dataset: {dataset}")
    
    initial_state = DSAState(dataset=dataset)
    if target:
        initial_state.planner_context.target_column = target
        
    final_state = asyncio.run(dsa_graph.ainvoke(initial_state))
    typer.echo(f"Execution finished. Status: {final_state.status}")

if __name__ == "__main__":
    app()
