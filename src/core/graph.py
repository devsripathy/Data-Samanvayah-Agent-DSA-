"""LangGraph workflow definition for DSA."""
from langgraph.graph import StateGraph, START, END
from src.core.state import DSAState
from src.agents.supervisor.node import supervisor_node
from src.agents.planner.node import planner_node
from src.agents.quality.node import quality_node
from src.agents.memory.node import memory_node
from src.agents.explorer.node import explorer_node
from src.agents.trainer.node import trainer_node
from src.agents.critic.node import critic_node

def route_supervisor(state: DSAState) -> str:
    """Determines the next node based on supervisor's decision."""
    return state.metadata.get("next_agent", "planner")

def build_dsa_graph() -> StateGraph:
    """Constructs and compiles the DSA LangGraph."""
    workflow = StateGraph(DSAState)

    # Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("quality", quality_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("explorer", explorer_node)
    workflow.add_node("trainer", trainer_node)
    workflow.add_node("critic", critic_node)

    # Define Edges
    workflow.add_edge(START, "memory")
    workflow.add_edge("memory", "quality")
    workflow.add_edge("quality", "supervisor")
    
    # Supervisor routes dynamically
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "planner": "planner",
            "explorer": "explorer",
            "critic": "critic",
            "end": END
        }
    )
    
    workflow.add_edge("planner", "explorer")
    workflow.add_edge("explorer", "trainer")
    workflow.add_edge("trainer", "critic")
    workflow.add_edge("critic", "supervisor") # Loop back for validation

    return workflow.compile()

# Global graph instance
dsa_graph = build_dsa_graph()
