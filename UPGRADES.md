# DSA Upgrades: LangGraph, LangSmith & Human-in-the-Loop Architecture

This document describes the comprehensive upgrades made to the Data Samanvayah Agent (DSA) to support production-grade features including persistent checkpointing, observability tracing, and human-in-the-loop workflows.

## Table of Contents

1. [Overview](#overview)
2. [Pydantic Serialization Improvements](#pydantic-serialization-improvements)
3. [LangSmith Integration](#langsmith-integration)
4. [Typer CLI with Session Management](#typer-cli-with-session-management)
5. [Human-in-the-Loop Architecture](#human-in-the-loop-architecture)
6. [Usage Examples](#usage-examples)
7. [API Reference](#api-reference)

---

## Overview

The DSA has been upgraded with the following production-ready features:

| Feature | Purpose | Status |
|---------|---------|--------|
| **Thread ID Management** | Persistent session tracking across restarts | ✓ Complete |
| **Typer CLI** | User-friendly command-line interface with `--session-id` support | ✓ Complete |
| **Pydantic Serialization** | Full serialization compatibility for checkpointing | ✓ Complete |
| **LangSmith Observability** | Trace agent decisions and execution flow | ✓ Complete |
| **Human-in-the-Loop** | Request human approval before Critic decisions | ✓ Complete |
| **SQLite/Postgres Checkpointing** | Persistent state management across sessions | ✓ Complete |

---

## Pydantic Serialization Improvements

### New Fields in DSAState

The `DSAState` model now includes:

```python
# Session Management for Checkpointing
session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
thread_id: str = Field(default_factory=lambda: f"thread_{uuid.uuid4().hex[:8]}")

# Human-in-the-Loop Support
pending_human_approval: bool = False
approval_request: str | None = None
```

### Enhanced Model Configuration

```python
model_config = ConfigDict(
    arbitrary_types_allowed=True,
    serialize_as_any=True,  # Better serialization support
)
```

### Serialization Support

New methods for checkpoint-safe serialization:

```python
@field_serializer('timestamp', when_used='json')
def serialize_timestamp(self, value: datetime) -> str:
    """Serialize datetime to ISO format string for JSON checkpointing."""
    return value.isoformat()

def model_dump_for_checkpoint(self) -> dict[str, Any]:
    """Returns a checkpoint-safe dictionary representation."""
    return self.model_dump(
        exclude_none=False,
        mode='json',
        serialize_as_any=True,
    )
```

### New Helper Methods

```python
def request_human_approval(self, request: str) -> None:
    """Requests human approval by setting the pending flag."""
    self.pending_human_approval = True
    self.approval_request = request
    self.status = "awaiting_approval"

def approve_and_continue(self) -> None:
    """Clears the human approval flags to allow workflow to continue."""
    self.pending_human_approval = False
    self.approval_request = None
    self.status = "approved"
```

---

## LangSmith Integration

### Configuration

Enable LangSmith in your `.env` file:

```env
OBSERVABILITY__LANGSMITH_ENABLED=true
OBSERVABILITY__LANGSMITH_API_KEY=ls_...your_key...
OBSERVABILITY__LANGSMITH_PROJECT=dsa-agent
```

Or in `config.yaml`:

```yaml
observability:
  langsmith_enabled: true
  langsmith_api_key: "ls_...your_key..."
  langsmith_project: "dsa-agent"
```

### New Observability Module

Located in `src/core/observability.py`:

#### Initialize Tracing

```python
from src.core.observability import init_observability

# Call once at application startup
init_observability()
```

#### Trace Agent Execution

```python
from src.core.observability import trace_agent_execution

async def agent_node(state: DSAState) -> dict:
    with trace_agent_execution(
        "planner",
        state.session_id,
        state.thread_id,
        metadata={"dataset": state.dataset}
    ) as trace_ctx:
        trace_ctx["input"] = state
        # ... agent logic ...
        trace_ctx["output"] = result
    return result
```

#### Trace Decision Points

```python
from src.core.observability import trace_decision_point

with trace_decision_point(
    "critic_approval",
    session_id,
    decision_data={"accuracy": 0.92, "f1": 0.88}
) as ctx:
    ctx["decision"] = "approved" if accuracy > 0.9 else "rejected"
    ctx["reasoning"] = f"Accuracy {accuracy} exceeds threshold"
```

#### Log Trace Events

```python
from src.core.observability import log_trace_event

log_trace_event(
    "retry_triggered",
    "supervisor",
    session_id,
    details={"retry_count": 2, "reason": "Low accuracy"}
)
```

### LangSmith Dashboard

View execution traces at: https://smith.langchain.com/

Key metrics tracked:
- **Agent Execution**: Start time, duration, success/failure
- **Decision Points**: Routing decisions, confidence scores
- **Retry Logic**: Number of retries, reasons for retries
- **Human Interventions**: HITL requests and approvals

---

## Typer CLI with Session Management

### New CLI Commands

#### 1. Run a New Session

```bash
dsa run data/sample.csv
```

Options:
```bash
dsa run data/sample.csv \
  --session-id abc123def456 \
  --thread-id thread_xyz789 \
  --enable-hitl \
  --sqlite
```

#### 2. List All Sessions

```bash
dsa list-sessions
```

Output:
```
Available Sessions
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Session ID     ┃ Timestamp                  ┃ Status     ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ abc123de...    │ 2024-07-09T15:22:10.123456 │ completed  │
│ xyz789ab...    │ 2024-07-09T14:18:45.654321 │ running    │
└────────────────┴────────────────────────────┴────────────┘
```

#### 3. Resume a Session

```bash
dsa resume abc123def456
```

Continue with a new dataset:
```bash
dsa resume abc123def456 --dataset data/new_dataset.csv
```

#### 4. Check Session Status

```bash
dsa status abc123def456
```

Output:
```
=== Session Status ===
Session ID: abc123def456
Thread ID: thread_xyz789
Status: awaiting_approval
Timestamp: 2024-07-09T15:22:10.123456+00:00
Completed Agents: memory, quality, supervisor, planner, explorer, trainer, critic
Logs: 42 entries
Errors: 0 entries

Pending Human Approval:
  Critic evaluation inconclusive (confidence: 0.68). 
  Issues found: Accuracy below 90% threshold
  Approve to continue or provide feedback?
```

### Session Storage

Sessions are stored in `./data/sessions/` as JSON files:

```json
{
  "session_id": "abc123def456",
  "timestamp": "2024-07-09T15:22:10.123456+00:00",
  "state": {
    "session_id": "abc123def456",
    "thread_id": "thread_xyz789",
    "status": "awaiting_approval",
    "dataset": "data/sample.csv",
    "completed_agents": ["memory", "quality", "supervisor", "planner", "explorer", "trainer", "critic"],
    "pending_human_approval": true,
    "approval_request": "Critic evaluation inconclusive...",
    ...
  }
}
```

---

## Human-in-the-Loop Architecture

### Design Pattern

The HITL implementation uses **interrupt-based routing** with the following flow:

```
Trainer → Critic Node (Evaluation) → Route Decision
                                      ↓
                        Is confidence >= threshold?
                        ↙                        ↘
                    YES (High Confidence)   NO (Low Confidence)
                    ↓                           ↓
            Deterministic Retry        Request Human Approval
            (Auto-retry or End)        (PAUSE WORKFLOW)
                                        ↓
                                Human Reviews & Decides
                                ↙             ↘
                            Approve        Reject
                            ↓              ↓
                        Continue      Modify & Retry
```

### Configuration in Code

```python
# Enable HITL with default settings
from src.core.graph import build_dsa_graph

graph = build_dsa_graph(enable_hitl=True)

# The graph is now compiled with interrupt_before=["critic"]
# This allows the workflow to pause before Critic execution
```

### Runtime Configuration

In state user_preferences:

```python
state.user_preferences.update({
    "enable_hitl": True,                    # Enable/disable HITL
    "hitl_confidence_threshold": 0.7,       # Confidence threshold for approval
    "max_retries": 3,                       # Max auto-retry attempts
})
```

### Decision Logic in `route_critic()`

```python
def route_critic(state: DSAState) -> Literal["supervisor", "end"]:
    """Routes based on critic evaluation with HITL support."""
    
    if state.critic_context.passed:
        return "supervisor"  # Proceed to memory store
    
    # Critic failed - check confidence
    confidence = state.critic_context.confidence
    enable_hitl = state.user_preferences.get("enable_hitl", True)
    threshold = state.user_preferences.get("hitl_confidence_threshold", 0.7)
    
    if enable_hitl and confidence < threshold:
        # Low confidence failure - request human review
        state.request_human_approval(
            f"Critic evaluation inconclusive (confidence: {confidence}). "
            f"Issues: {', '.join(state.critic_context.issues_found)}"
        )
        return "supervisor"
    
    # High confidence failure - retry without human intervention
    state.metadata["next_agent"] = "retry"
    return "supervisor"
```

### Handling HITL in Your Code

```python
# After execution, check if approval is needed
from src.core.graph import build_dsa_graph

graph = build_dsa_graph(enable_hitl=True)
config = {"configurable": {"thread_id": state.thread_id}}

# Stream execution
for event in graph.stream(state, config=config):
    # Check if workflow is waiting for approval
    if state.pending_human_approval:
        print(f"Awaiting approval: {state.approval_request}")
        
        # Get human decision (from API, CLI, etc.)
        human_decision = input("Approve? (yes/no): ")
        
        if human_decision.lower() == "yes":
            state.approve_and_continue()
            # Resume with updated state
            final = graph.invoke(state, config=config)
        else:
            state.status = "rejected"
```

### Critic Node with Confidence Score

Update `src/agents/critic/node.py`:

```python
async def critic_node(state: DSAState) -> dict:
    """Evaluates training results with confidence scoring."""
    state.update_status("critiquing", "critic")
    state.append_log("Evaluating model performance.")
    
    metrics = state.training_results.metrics
    accuracy = metrics.get("accuracy", 0)
    f1 = metrics.get("f1", 0)
    
    # Determine pass/fail
    passed = accuracy > 0.90
    
    # Calculate confidence (example: based on metric consistency)
    confidence = (accuracy + f1) / 2  # Simple average
    
    context = CriticContext(
        passed=passed,
        confidence=confidence,  # Add this field!
        issues_found=[] if passed else ["Accuracy below 90% threshold"],
        retry_reason="Improve feature engineering" if not passed else None,
        recommendations=["Try hyperparameter tuning"] if not passed else []
    )
    
    state.append_log(f"Critique: {'passed' if passed else 'failed'} (confidence={confidence:.2f})")
    return {"critic_context": context}
```

Don't forget to add the `confidence` field to `CriticContext`:

```python
class CriticContext(BaseModel):
    """Context and evaluation results from the Critic agent."""
    
    passed: bool = False
    confidence: float = 0.0  # Add this field
    issues_found: list[str] = Field(default_factory=list)
    retry_reason: str | None = None
    retry_count: int = 0
    recommendations: list[str] = Field(default_factory=list)
```

---

## Usage Examples

### Example 1: Basic Execution with Session ID

```bash
# Create a new session
dsa run data/sample.csv --session-id my_first_run

# Output:
# Starting DSA pipeline...
# ✓ memory: running
# ✓ quality: running
# ✓ supervisor: running
# ✓ planner: running
# ✓ explorer: running
# ✓ trainer: running
# ✓ critic: running
# ✓ memory_store: running
# 
# === Execution Summary ===
# Session ID: my_first_run
# Thread ID: thread_abc12345
# Status: completed
# Completed Agents: memory, quality, supervisor, planner, explorer, trainer, critic, memory_store
```

### Example 2: Resume with Different Dataset

```bash
# Check previous session
dsa status my_first_run

# Resume with new dataset
dsa resume my_first_run --dataset data/new_dataset.csv

# Session uses the previous state but with new dataset
```

### Example 3: Enable HITL for Approval Workflow

```bash
# Run with HITL enabled
dsa run data/sample.csv --session-id hitl_session --enable-hitl

# If Critic confidence is low:
# === Execution Summary ===
# Status: awaiting_approval
# Pending Human Approval:
#   Critic evaluation inconclusive (confidence: 0.65).
#   Issues found: Accuracy below 90% threshold
#   Approve to continue or provide feedback?

# Manually approve
dsa resume hitl_session  # Would continue after approval

# Or write a Python script:
from src.core.state import DSAState
from main import load_session_checkpoint, execute_pipeline

state = load_session_checkpoint("hitl_session")
if state.pending_human_approval:
    print(f"Review: {state.approval_request}")
    state.approve_and_continue()  # Or modify state and retry
    # Resume execution...
```

### Example 4: With SQLite Checkpointing

```bash
# Use SQLite for persistent storage
dsa run data/sample.csv --session-id prod_run --sqlite

# Checkpoints saved to: ./data/checkpoints/prod_run.db
# Session metadata saved to: ./data/sessions/prod_run.json

# Pause and resume without data loss
dsa resume prod_run
```

### Example 5: Enable LangSmith Observability

```bash
# In .env file:
OBSERVABILITY__LANGSMITH_ENABLED=true
OBSERVABILITY__LANGSMITH_API_KEY=ls_...

# Run session
dsa run data/sample.csv --session-id traced_run

# View traces at: https://smith.langchain.com/projects/dsa-agent
# Traces show:
# - Agent execution flow with timing
# - Decision points and routing
# - Retry logic and reasons
# - Human intervention requests
# - Error tracking and debugging info
```

---

## API Reference

### DSAState Methods

#### `request_human_approval(request: str) -> None`

Requests human approval by setting pending flags.

**Parameters:**
- `request` (str): Message describing what needs approval

**Example:**
```python
state.request_human_approval(
    "Accuracy (0.88) is below optimal threshold. Approve retry?"
)
```

#### `approve_and_continue() -> None`

Clears approval flags to allow workflow to continue.

**Example:**
```python
if state.pending_human_approval:
    state.approve_and_continue()
```

#### `model_dump_for_checkpoint() -> dict[str, Any]`

Returns checkpoint-safe dictionary representation.

**Example:**
```python
checkpoint_data = state.model_dump_for_checkpoint()
# Safe to serialize to JSON
```

### Observability Functions

#### `trace_agent_execution(agent_name, session_id, thread_id, metadata=None)`

Context manager for tracing agent execution.

**Parameters:**
- `agent_name` (str): Name of agent (e.g., "planner")
- `session_id` (str): Session identifier
- `thread_id` (str): Thread identifier
- `metadata` (dict, optional): Additional metadata

**Yields:**
- Dict with "input" and "output" keys

#### `trace_decision_point(decision_name, session_id, decision_data=None)`

Context manager for tracing decision points.

**Parameters:**
- `decision_name` (str): Decision point name
- `session_id` (str): Session identifier
- `decision_data` (dict, optional): Input data for decision

**Yields:**
- Dict with "decision" and "reasoning" keys

#### `log_trace_event(event_type, agent_name, session_id, details=None)`

Log a custom event to the trace.

**Parameters:**
- `event_type` (str): Type of event
- `agent_name` (str): Agent name
- `session_id` (str): Session identifier
- `details` (dict, optional): Additional event details

### Graph Functions

#### `build_dsa_graph(checkpointer=None, interrupt_before=None, interrupt_after=None, enable_hitl=True)`

Builds DSA graph with configuration.

**Parameters:**
- `checkpointer`: LangGraph checkpointer (defaults to MemorySaver)
- `interrupt_before`: List of nodes to interrupt before
- `interrupt_after`: List of nodes to interrupt after
- `enable_hitl` (bool): Enable human-in-the-loop

**Returns:**
- Compiled StateGraph instance

---

## Troubleshooting

### Session Not Loading

```bash
# Check available sessions
dsa list-sessions

# Verify session file exists
ls ./data/sessions/

# Check session details
dsa status <session-id>
```

### LangSmith Traces Not Appearing

1. Verify API key in `.env` or `config.yaml`
2. Check that `OBSERVABILITY__LANGSMITH_ENABLED=true`
3. Look for errors in logs:
   ```python
   logger.debug("LangSmith tracing disabled")
   ```

### HITL Not Triggering

1. Verify `enable_hitl=True` in user_preferences
2. Check that `CriticContext.confidence` is populated
3. Verify threshold: `state.user_preferences.get("hitl_confidence_threshold", 0.7)`

---

## Next Steps

1. **Deploy observability**: Set up LangSmith project for production tracing
2. **Implement custom HITL UI**: Create web interface for approvals instead of CLI
3. **Add persistence layer**: Implement database models for long-term session storage
4. **Enhance monitoring**: Add metrics for agent success rates, execution times, etc.
5. **Production testing**: Run load tests with concurrent sessions

---

For more information or issues, refer to:
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Typer Documentation](https://typer.tiangolo.com/)
