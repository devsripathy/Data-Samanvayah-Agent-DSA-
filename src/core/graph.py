"""
LangGraph workflow definition for the Data Samanvayah Agent (DSA).

This module constructs the complete multi-agent orchestration graph with 
support for checkpointing, streaming, human-in-the-loop interrupts, 
time travel, and dynamic routing. It includes Mermaid diagram generation 
for visualization and documentation.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.critic.node import critic_node
from src.agents.explorer.node import explorer_node
from src.agents.memory.node import memory_node, memory_store_node
from src.agents.planner.node import planner_node
from src.agents.quality.node import quality_node
from src.agents.supervisor.node import supervisor_node
from src.agents.trainer.node import trainer_node
from src.core.state import DSAState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------

def route_supervisor(state: DSAState) -> Literal["planner", "explorer", "critic", "memory_store", "end"]:
    """
    Determines the next node based on supervisor's decision and state.
    
    Args:
        state: Current DSA state containing metadata and execution context.
        
    Returns:
        String identifier of the next node to execute.
    """
    next_agent = state.metadata.get("next_agent", "planner")
    
    # Handle retry logic
    if next_agent == "retry":
        if state.retry_counter < state.user_preferences.get("max_retries", 3):
            logger.info(f"Routing to planner for retry attempt {state.retry_counter + 1}")
            return "planner"
        else:
            logger.warning("Max retries exceeded. Ending workflow.")
            return "end"
    
    # Handle normal flow
    if next_agent == "memory_store":
        return "memory_store"
    elif next_agent in ["planner", "explorer", "critic"]:
        return next_agent
    else:
        return "end"


def route_critic(state: DSAState) -> Literal["supervisor", "end"]:
    """
    Routes based on critic evaluation results.
    
    Args:
        state: Current DSA state with critic context.
        
    Returns:
        'supervisor' if retry needed, 'end' if workflow complete.
    """
    if state.critic_context and state.critic_context.passed:
        logger.info("Critic passed. Routing to memory store.")
        return "supervisor"
    else:
        logger.info("Critic failed. Routing back to supervisor for retry decision.")
        state.metadata["next_agent"] = "retry"
        return "supervisor"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_dsa_graph(
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> StateGraph:
    """
    Constructs and compiles the DSA LangGraph with full enterprise features.
    
    Args:
        checkpointer: Optional checkpoint saver for persistence and time travel.
                     Defaults to MemorySaver if None.
        interrupt_before: List of node names to interrupt before execution.
        interrupt_after: List of node names to interrupt after execution.
        
    Returns:
        Compiled LangGraph StateGraph instance.
    """
    # Initialize checkpointer for persistence and time travel
    if checkpointer is None:
        checkpointer = MemorySaver()
        logger.info("Using in-memory checkpointer. State will not persist across restarts.")
    
    # Create the workflow
    workflow = StateGraph(DSAState)

    # -------------------------------------------------------------------------
    # Add Nodes
    # -------------------------------------------------------------------------
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("quality", quality_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("explorer", explorer_node)
    workflow.add_node("trainer", trainer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("memory_store", memory_store_node)

    # -------------------------------------------------------------------------
    # Define Edges
    # -------------------------------------------------------------------------
    
    # Initial flow: START -> Memory -> Quality -> Supervisor
    workflow.add_edge(START, "memory")
    workflow.add_edge("memory", "quality")
    workflow.add_edge("quality", "supervisor")
    
    # Supervisor routes dynamically based on state
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "planner": "planner",
            "explorer": "explorer",
            "critic": "critic",
            "memory_store": "memory_store",
            "end": END,
        },
    )
    
    # Execution flow: Planner -> Explorer -> Trainer -> Critic
    workflow.add_edge("planner", "explorer")
    workflow.add_edge("explorer", "trainer")
    workflow.add_edge("trainer", "critic")
    
    # Critic routes back to supervisor for retry or to end
    workflow.add_conditional_edges(
        "critic",
        route_critic,
        {
            "supervisor": "supervisor",
            "end": END,
        },
    )
    
    # Memory store always ends the workflow
    workflow.add_edge("memory_store", END)

    # -------------------------------------------------------------------------
    # Compile with Features
    # -------------------------------------------------------------------------
    compiled_graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )
    
    logger.info("DSA graph compiled successfully with checkpointing and interrupt support.")
    return compiled_graph


# ---------------------------------------------------------------------------
# Mermaid Diagram Generation
# ---------------------------------------------------------------------------

def generate_mermaid_diagram(graph: StateGraph | None = None) -> str:
    """
    Generates a Mermaid diagram string for visualization.
    
    Args:
        graph: Optional compiled graph. Uses default if None.
        
    Returns:
        Mermaid diagram as a string.
    """
    if graph is None:
        graph = build_dsa_graph()
    
    try:
        mermaid_code = graph.get_graph().draw_mermaid()
        logger.info("Mermaid diagram generated successfully.")
        return mermaid_code
    except Exception as e:
        logger.error(f"Failed to generate Mermaid diagram: {e}")
        return ""


def save_mermaid_diagram(output_path: str = "docs/dsa_graph.mmd") -> None:
    """
    Saves the Mermaid diagram to a file.
    
    Args:
        output_path: Path to save the .mmd file.
    """
    import os
    from pathlib import Path
    
    mermaid_code = generate_mermaid_diagram()
    if mermaid_code:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        logger.info(f"Mermaid diagram saved to {output_path}")


# ---------------------------------------------------------------------------
# Global Graph Instance
# ---------------------------------------------------------------------------

# Default graph with in-memory checkpointing
dsa_graph = build_dsa_graph()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_graph_with_sqlite_persistence(db_path: str = "./data/dsa_checkpoints.db") -> StateGraph:
    """
    Creates a graph with SQLite-based persistence for production use.
    
    Args:
        db_path: Path to the SQLite database file.
        
    Returns:
        Compiled graph with SQLite checkpointer.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        
        checkpointer = SqliteSaver.from_conn_string(db_path)
        logger.info(f"Using SQLite checkpointer at {db_path}")
        return build_dsa_graph(checkpointer=checkpointer)
    except ImportError:
        logger.warning("SQLite checkpointer not available. Falling back to memory.")
        return build_dsa_graph()


def get_graph_with_postgres_persistence(connection_string: str) -> StateGraph:
    """
    Creates a graph with PostgreSQL-based persistence for production use.
    
    Args:
        connection_string: PostgreSQL connection string.
        
    Returns:
        Compiled graph with Postgres checkpointer.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        
        checkpointer = PostgresSaver.from_conn_string(connection_string)
        logger.info("Using PostgreSQL checkpointer")
        return build_dsa_graph(checkpointer=checkpointer)
    except ImportError:
        logger.warning("PostgreSQL checkpointer not available. Falling back to memory.")
        return build_dsa_graph()


# ---------------------------------------------------------------------------
# Streaming Support
# ---------------------------------------------------------------------------

async def stream_graph(
    graph: StateGraph | None = None,
    initial_state: DSAState | None = None,
    config: dict[str, Any] | None = None,
):
    """
    Streams the graph execution for real-time monitoring.
    
    Args:
        graph: Compiled graph to stream. Uses default if None.
        initial_state: Initial state to pass to the graph.
        config: Optional configuration including thread_id for checkpointing.
        
    Yields:
        Tuples of (node_name, state_update) for each step.
    """
    if graph is None:
        graph = dsa_graph
    
    if initial_state is None:
        initial_state = DSAState()
    
    if config is None:
        config = {"configurable": {"thread_id": "default"}}
    
    async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
        yield event


# ---------------------------------------------------------------------------
# Time Travel Support
# ---------------------------------------------------------------------------

def get_state_at_checkpoint(
    graph: StateGraph | None = None,
    thread_id: str = "default",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """
    Retrieves the state at a specific checkpoint for time travel.
    
    Args:
        graph: Compiled graph with checkpointer.
        thread_id: Thread identifier for the execution.
        checkpoint_id: Optional specific checkpoint ID. If None, returns latest.
        
    Returns:
        State dictionary at the specified checkpoint.
    """
    if graph is None:
        graph = dsa_graph
    
    config = {"configurable": {"thread_id": thread_id}}
    
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id
    
    try:
        state = graph.get_state(config)
        return state.values if state else {}
    except Exception as e:
        logger.error(f"Failed to retrieve checkpoint state: {e}")
        return {}


def list_checkpoints(
    graph: StateGraph | None = None,
    thread_id: str = "default",
) -> list[dict[str, Any]]:
    """
    Lists all checkpoints for a given thread.
    
    Args:
        graph: Compiled graph with checkpointer.
        thread_id: Thread identifier.
        
    Returns:
        List of checkpoint metadata dictionaries.
    """
    if graph is None:
        graph = dsa_graph
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        checkpoints = list(graph.get_state_history(config))
        return [
            {
                "checkpoint_id": cp.config["configurable"].get("checkpoint_id"),
                "timestamp": cp.created_at,
                "node": cp.next,
            }
            for cp in checkpoints
        ]
    except Exception as e:
        logger.error(f"Failed to list checkpoints: {e}")
        return []
