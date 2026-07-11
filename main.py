"""
Main execution script for the Data Samanvayah Agent (DSA).

Provides a CLI interface with session management, checkpointing support,
LangSmith observability integration, and human-in-the-loop workflows.

Usage:
    dsa run --dataset data/sample.csv                    # New session
    dsa run --dataset data/sample.csv --session-id xyz   # Resume session
    dsa list-sessions                                    # Show all sessions
    dsa resume --session-id xyz                          # Resume specific session
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.core.config import get_settings
from src.core.graph import build_dsa_graph, get_graph_with_sqlite_persistence
from src.core.observability import init_observability, log_trace_event
from src.core.state import DSAState
from src.utils.logger import get_logger

app = typer.Typer(
    help="Data Samanvayah Agent (DSA) - Enterprise-grade Autonomous Data Science Agent",
    rich_markup_mode="rich",
)
console = Console()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------

SESSION_STORE_DIR = Path("./data/sessions")


def ensure_session_dir() -> None:
    """Ensures the session store directory exists."""
    SESSION_STORE_DIR.mkdir(parents=True, exist_ok=True)


def get_session_path(session_id: str) -> Path:
    """Gets the file path for a session's checkpoint."""
    return SESSION_STORE_DIR / f"{session_id}.json"


def save_session_checkpoint(session_id: str, state: DSAState) -> None:
    """Saves a session checkpoint to disk."""
    ensure_session_dir()
    session_path = get_session_path(session_id)
    try:
        checkpoint_data = {
            "session_id": session_id,
            "state": state.model_dump_for_checkpoint(),
            "timestamp": state.timestamp.isoformat(),
        }
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        logger.info(f"Session checkpoint saved: {session_path}")
    except Exception as e:
        logger.error(f"Failed to save session checkpoint: {e}")


def load_session_checkpoint(session_id: str) -> DSAState | None:
    """Loads a session checkpoint from disk."""
    ensure_session_dir()
    session_path = get_session_path(session_id)
    
    if not session_path.exists():
        logger.warning(f"Session checkpoint not found: {session_path}")
        return None
    
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        
        state = DSAState(**checkpoint_data["state"])
        logger.info(f"Session checkpoint loaded: {session_path}")
        return state
    except Exception as e:
        logger.error(f"Failed to load session checkpoint: {e}")
        return None


def list_all_sessions() -> list[dict]:
    """Lists all available sessions."""
    ensure_session_dir()
    sessions = []
    
    for session_file in SESSION_STORE_DIR.glob("*.json"):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "session_id": data.get("session_id", session_file.stem),
                "timestamp": data.get("timestamp", "unknown"),
                "status": data.get("state", {}).get("status", "unknown"),
            })
        except Exception as e:
            logger.error(f"Error reading session {session_file}: {e}")
    
    return sessions


# ---------------------------------------------------------------------------
# Core Execution Functions
# ---------------------------------------------------------------------------

async def execute_pipeline(
    dataset_path: str,
    session_id: str | None = None,
    thread_id: str | None = None,
    enable_hitl: bool = True,
    use_sqlite: bool = False,
) -> DSAState:
    """
    Initializes state and runs the DSA pipeline.
    
    Args:
        dataset_path: Path to the dataset file.
        session_id: Optional session ID (auto-generated if None).
        thread_id: Optional thread ID (auto-generated if None).
        enable_hitl: Whether to enable human-in-the-loop workflows.
        use_sqlite: Whether to use SQLite for checkpointing.
        
    Returns:
        The final state after pipeline execution.
    """
    # Initialize observability
    init_observability()
    
    # Try to load existing session
    initial_state: DSAState | None = None
    if session_id:
        initial_state = load_session_checkpoint(session_id)
    
    # Create new state if not resuming
    if initial_state is None:
        initial_state = DSAState(
            dataset=dataset_path,
            status="starting",
        )
        if session_id:
            initial_state.session_id = session_id
        if thread_id:
            initial_state.thread_id = thread_id
    
    # Log session start
    initial_state.append_log(f"Pipeline initialized with session_id={initial_state.session_id}")
    logger.info(
        f"Starting DSA execution: session={initial_state.session_id}, "
        f"thread={initial_state.thread_id}, dataset={dataset_path}"
    )
    
    # Set user preferences
    initial_state.user_preferences.update({
        "enable_hitl": enable_hitl,
        "dataset_path": dataset_path,
    })
    
    # Select graph with appropriate checkpointer
    if use_sqlite:
        db_path = f"./data/checkpoints/{initial_state.session_id}.db"
        graph = get_graph_with_sqlite_persistence(db_path)
        logger.info(f"Using SQLite checkpointer: {db_path}")
    else:
        graph = build_dsa_graph(enable_hitl=enable_hitl)
    
    # Prepare execution config with thread_id for checkpointing
    config = {
        "configurable": {
            "thread_id": initial_state.thread_id,
        }
    }
    
    # Execute the graph
    try:
        log_trace_event(
            "pipeline_start",
            "system",
            initial_state.session_id,
            {
                "thread_id": initial_state.thread_id,
                "enable_hitl": enable_hitl,
            },
        )
        
        # Stream execution for real-time updates
        console.print("[bold cyan]Starting DSA pipeline...[/bold cyan]")
        async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_state in event.items():
                if node_name != "__start__":
                    status = node_state.get('status', 'running')
                    console.print(f"[green]*[/green] {node_name}: {status}")
        
        # Get final state from checkpoint
        final_state_obj = graph.get_state(config)
        final_state = DSAState(**final_state_obj.values) if final_state_obj else initial_state
        
        logger.info(f"Pipeline completed with status: {final_state.status}")
        console.print("[bold green]Pipeline execution completed![/bold green]")
        
        # Save checkpoint
        save_session_checkpoint(final_state.session_id, final_state)
        
        log_trace_event(
            "pipeline_complete",
            "system",
            final_state.session_id,
            {"status": final_state.status},
        )
        
        return final_state
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        initial_state.status = "failed"
        initial_state.add_error(str(e))
        save_session_checkpoint(initial_state.session_id, initial_state)
        
        log_trace_event(
            "pipeline_error",
            "system",
            initial_state.session_id,
            {"error": str(e)},
        )
        
        console.print(f"[bold red]✗ Pipeline failed: {e}[/bold red]")
        raise


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    dataset: str = typer.Argument(..., help="Path to the dataset file"),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        help="Optional session ID to resume (auto-generated if not provided)",
    ),
    thread_id: Optional[str] = typer.Option(
        None,
        "--thread-id",
        help="Optional thread ID for execution continuity",
    ),
    enable_hitl: bool = typer.Option(
        True,
        "--enable-hitl/--disable-hitl",
        help="Enable human-in-the-loop for Critic agent",
    ),
    use_sqlite: bool = typer.Option(
        False,
        "--sqlite",
        help="Use SQLite for persistent checkpointing",
    ),
) -> None:
    """
    Run the DSA pipeline on a dataset.
    
    Example:
        dsa run data/sample.csv                           # New session
        dsa run data/sample.csv --session-id abc123       # Resume session
        dsa run data/sample.csv --enable-hitl             # With HITL enabled
    """
    try:
        # Validate dataset path
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            console.print(f"[bold red]Error: Dataset not found: {dataset}[/bold red]")
            raise typer.Exit(code=1)
        
        # Execute pipeline
        final_state = asyncio.run(
            execute_pipeline(
                dataset_path=str(dataset_path.absolute()),
                session_id=session_id,
                thread_id=thread_id,
                enable_hitl=enable_hitl,
                use_sqlite=use_sqlite,
            )
        )
        
        # Display results
        console.print("\n[bold cyan]=== Execution Summary ===[/bold cyan]")
        console.print(f"Session ID: [yellow]{final_state.session_id}[/yellow]")
        console.print(f"Thread ID: [yellow]{final_state.thread_id}[/yellow]")
        console.print(f"Status: [yellow]{final_state.status}[/yellow]")
        console.print(f"Completed Agents: {', '.join(final_state.completed_agents)}")
        
        if final_state.errors:
            console.print(f"\n[bold red]Errors:[/bold red]")
            for error in final_state.errors:
                console.print(f"  - {error}")
        
        if final_state.critic_context:
            console.print(f"\n[bold cyan]Critic Evaluation:[/bold cyan]")
            console.print(f"  Passed: {final_state.critic_context.passed}")
            if final_state.critic_context.issues_found:
                console.print(f"  Issues: {', '.join(final_state.critic_context.issues_found)}")
        
    except Exception as e:
        console.print(f"[bold red]Execution failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def list_sessions() -> None:
    """
    List all available sessions with their status.
    
    Example:
        dsa list-sessions
    """
    sessions = list_all_sessions()
    
    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return
    
    table = Table(title="Available Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Timestamp", style="magenta")
    table.add_column("Status", style="green")
    
    for session in sessions:
        table.add_row(
            session["session_id"][:8] + "...",
            session["timestamp"],
            session["status"],
        )
    
    console.print(table)


@app.command()
def resume(
    session_id: str = typer.Argument(..., help="Session ID to resume"),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        help="Optional dataset path (uses original if not provided)",
    ),
) -> None:
    """
    Resume a previous session.
    
    Example:
        dsa resume abc123def456
        dsa resume abc123def456 --dataset data/new_dataset.csv
    """
    try:
        # Load session checkpoint
        state = load_session_checkpoint(session_id)
        if state is None:
            console.print(f"[bold red]Session not found: {session_id}[/bold red]")
            raise typer.Exit(code=1)
        
        # Override dataset if provided
        if dataset:
            dataset_path = dataset
        else:
            dataset_path = state.user_preferences.get("dataset_path")
        
        if not dataset_path:
            console.print("[bold red]No dataset specified. Use --dataset to provide one.[/bold red]")
            raise typer.Exit(code=1)
        
        console.print(f"[bold cyan]Resuming session: {session_id}[/bold cyan]")
        
        # Resume execution
        final_state = asyncio.run(
            execute_pipeline(
                dataset_path=dataset_path,
                session_id=session_id,
                thread_id=state.thread_id,
                enable_hitl=state.user_preferences.get("enable_hitl", True),
            )
        )
        
        console.print(f"\n[bold green]✓ Session resumed and completed![/bold green]")
        console.print(f"Final Status: {final_state.status}")
        
    except Exception as e:
        console.print(f"[bold red]Failed to resume session: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def status(
    session_id: str = typer.Argument(..., help="Session ID to check"),
) -> None:
    """
    Check the status of a session.
    
    Example:
        dsa status abc123def456
    """
    try:
        state = load_session_checkpoint(session_id)
        if state is None:
            console.print(f"[bold red]Session not found: {session_id}[/bold red]")
            raise typer.Exit(code=1)
        
        console.print(f"[bold cyan]=== Session Status ===[/bold cyan]")
        console.print(f"Session ID: [yellow]{state.session_id}[/yellow]")
        console.print(f"Thread ID: [yellow]{state.thread_id}[/yellow]")
        console.print(f"Status: [yellow]{state.status}[/yellow]")
        console.print(f"Timestamp: [yellow]{state.timestamp.isoformat()}[/yellow]")
        console.print(f"Completed Agents: {', '.join(state.completed_agents) or 'None'}")
        console.print(f"Logs: {len(state.logs)} entries")
        console.print(f"Errors: {len(state.errors)} entries")
        
        if state.pending_human_approval:
            console.print(f"\n[bold red]Pending Human Approval:[/bold red]")
            console.print(f"  {state.approval_request}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Legacy Support
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        help="Show version information",
    ),
) -> None:
    """Data Samanvayah Agent - Autonomous Data Science System"""
    if version:
        console.print("[bold cyan]DSA v0.1.0[/bold cyan]")
        raise typer.Exit()
    
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


if __name__ == "__main__":
    app()
