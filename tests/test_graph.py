"""Tests for DSA Graph and State."""
import pytest
from src.core.state import DSAState
from src.core.graph import build_dsa_graph

def test_state_initialization():
    """Test that DSAState initializes with correct defaults."""
    state = DSAState(dataset="test.csv")
    assert state.status == "initialized"
    assert state.retry_counter == 0
    assert state.execution_id is not None

def test_state_helpers():
    """Test DSAState helper methods."""
    state = DSAState(dataset="test.csv")
    state.append_log("Test log", level="DEBUG")
    assert len(state.logs) == 1
    assert state.logs[0]["level"] == "DEBUG"
    
    state.mark_completed("memory")
    assert "memory" in state.completed_agents

def test_graph_compilation():
    """Test that the LangGraph compiles without errors."""
    graph = build_dsa_graph()
    assert graph is not None
