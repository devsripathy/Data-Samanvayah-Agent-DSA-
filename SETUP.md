# DSA Setup & Getting Started Guide

This guide walks you through setting up the upgraded Data Samanvayah Agent with all new features enabled.

## Quick Start

### 1. Install Dependencies

```bash
# Install base dependencies
pip install -e .

# Install with observability support
pip install -e ".[observability]"
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# LLM Configuration
LLM__PROVIDER=openai
LLM__MODEL=gpt-4o
LLM__API_KEY=sk_...your_openai_key...
LLM__TEMPERATURE=0.2
LLM__MAX_TOKENS=4096

# Database Configuration
DATABASE__TYPE=sqlite
DATABASE__SQLITE_PATH=./data/dsa.db

# LangSmith Observability
OBSERVABILITY__LANGSMITH_ENABLED=true
OBSERVABILITY__LANGSMITH_API_KEY=ls_...your_langsmith_key...
OBSERVABILITY__LANGSMITH_PROJECT=dsa-agent

# Logging
LOGGING__LEVEL=INFO
```

Or create a `config.yaml` file:

```yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: "sk_..."
  temperature: 0.2
  max_tokens: 4096

database:
  type: sqlite
  sqlite_path: ./data/dsa.db

observability:
  langsmith_enabled: true
  langsmith_api_key: "ls_..."
  langsmith_project: dsa-agent

logging:
  level: INFO
```

### 3. Create Data Directory

```bash
mkdir -p data/sessions
mkdir -p data/checkpoints
mkdir -p artifacts/{models,logs,reports}
```

### 4. Run Your First Session

```bash
# New session
dsa run data/sample.csv

# Named session for tracking
dsa run data/sample.csv --session-id my_first_experiment

# With all features
dsa run data/sample.csv \
  --session-id full_featured \
  --enable-hitl \
  --sqlite
```

### 5. Monitor Execution

```bash
# List all sessions
dsa list-sessions

# Check specific session status
dsa status my_first_experiment

# Resume a session
dsa resume my_first_experiment
```

---

## Feature Configuration

### Enable LangSmith Observability

#### Step 1: Get API Key

1. Go to [LangSmith](https://smith.langchain.com/)
2. Create an account or sign in
3. Generate an API key from settings
4. Copy the key (starts with `ls_`)

#### Step 2: Configure DSA

Add to `.env`:
```env
OBSERVABILITY__LANGSMITH_ENABLED=true
OBSERVABILITY__LANGSMITH_API_KEY=ls_...your_key...
OBSERVABILITY__LANGSMITH_PROJECT=dsa-agent
```

#### Step 3: Verify

```bash
# Run a session
dsa run data/sample.csv --session-id traced_run

# Check LangSmith dashboard
# https://smith.langchain.com/projects/dsa-agent
```

You should see traces appear in real-time as the workflow executes.

### Enable Human-in-the-Loop

#### Step 1: Run with HITL

```bash
dsa run data/sample.csv --enable-hitl
```

#### Step 2: Configure Threshold

In your code or config:

```python
state.user_preferences.update({
    "enable_hitl": True,
    "hitl_confidence_threshold": 0.7,  # Approval needed if confidence < 0.7
    "max_retries": 3,
})
```

#### Step 3: Handle Approvals

```bash
# Check if workflow is waiting
dsa status <session-id>

# If pending approval shown:
# - Review the approval request
# - Manually approve by resuming
dsa resume <session-id>
```

Or programmatically:

```python
from src.core.state import DSAState
from main import load_session_checkpoint, execute_pipeline

state = load_session_checkpoint("my_session")

if state.pending_human_approval:
    print(f"Approval needed: {state.approval_request}")
    
    # Manual review and approval
    user_input = input("Approve? (yes/no): ")
    
    if user_input.lower() == "yes":
        state.approve_and_continue()
        # Resume execution
```

### Enable SQLite Checkpointing

```bash
# Run with SQLite persistence
dsa run data/sample.csv --session-id persistent_run --sqlite

# Checkpoints saved to:
# - ./data/checkpoints/persistent_run.db
# - ./data/sessions/persistent_run.json
```

### Enable All Features Together

```bash
dsa run data/sample.csv \
  --session-id production_run \
  --thread-id thread_prod_001 \
  --enable-hitl \
  --sqlite
```

---

## Development Setup

### Install Development Dependencies

```bash
pip install -e ".[dev,observability]"
```

### Run Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

### Local LangSmith Development

For offline development, you can skip LangSmith:

```env
OBSERVABILITY__LANGSMITH_ENABLED=false
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   DSA Main (Typer CLI)                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Session Management     │  Checkpointing   │  LangSmith │
│  ├─ run               │  ├─ Memory       │  ├─ Traces │
│  ├─ resume            │  ├─ SQLite       │  ├─ Events │
│  ├─ status            │  └─ Postgres     │  └─ Metrics│
│  └─ list-sessions     │                  │             │
│                       │                  │             │
└──────────────────────┬─────────────────┬────────────────┘
                       │                 │
                       ▼                 ▼
            ┌──────────────────┐  ┌──────────────────┐
            │  LangGraph State │  │  Observability   │
            │  (DSAState)      │  │  (Tracing)       │
            └──────────────────┘  └──────────────────┘
                       │                 │
                       └────────┬────────┘
                                ▼
                    ┌──────────────────────┐
                    │  Agent Nodes         │
                    ├──────────────────────┤
                    │ ├─ Supervisor       │
                    │ ├─ Planner          │
                    │ ├─ Explorer         │
                    │ ├─ Trainer          │
                    │ ├─ Critic (HITL!)   │
                    │ ├─ Quality          │
                    │ └─ Memory           │
                    └──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Checkpointed State  │
                    │  Saved Checkpoint    │
                    └──────────────────────┘
```

---

## Common Workflows

### Workflow 1: Quick Experimentation

```bash
# Run multiple experiments with different datasets
dsa run data/dataset_v1.csv --session-id exp_v1
dsa run data/dataset_v2.csv --session-id exp_v2
dsa run data/dataset_v3.csv --session-id exp_v3

# Compare results
dsa status exp_v1
dsa status exp_v2
dsa status exp_v3

# List all experiments
dsa list-sessions
```

### Workflow 2: Production Deployment with Observability

```bash
# Enable full observability for production
export OBSERVABILITY__LANGSMITH_ENABLED=true
export OBSERVABILITY__LANGSMITH_API_KEY=ls_prod_key

# Run production session with persistence
dsa run data/production_dataset.csv \
  --session-id prod_run_2024 \
  --sqlite

# Monitor on LangSmith dashboard
# View traces, errors, latencies in real-time
```

### Workflow 3: Human Review & Approval

```bash
# Run with HITL enabled
dsa run data/sensitive_dataset.csv \
  --session-id review_required \
  --enable-hitl

# Workflow pauses at Critic if confidence is low
# Get notification: Pending Human Approval

# Review the state
dsa status review_required

# Make an informed decision
dsa resume review_required

# Or modify and retry
python <<'EOF'
from main import load_session_checkpoint

state = load_session_checkpoint("review_required")

# Modify based on review feedback
state.user_preferences["max_retries"] = 5

# Approve and continue
state.approve_and_continue()
EOF
```

### Workflow 4: Debugging Failed Sessions

```bash
# Check what went wrong
dsa status failed_session

# Load state and inspect
python <<'EOF'
from main import load_session_checkpoint

state = load_session_checkpoint("failed_session")

print("Last 5 logs:")
for log in state.logs[-5:]:
    print(f"  [{log['level']}] {log['message']}")

print("\nErrors:")
for error in state.errors:
    print(f"  - {error}")

print(f"\nCurrent agent: {state.current_agent}")
print(f"Completed agents: {state.completed_agents}")
EOF

# Modify state if needed and resume
dsa resume failed_session --dataset data/corrected_dataset.csv
```

---

## Troubleshooting

### Issue: "Session not found"

```bash
# Check available sessions
dsa list-sessions

# Verify session file exists
ls -la data/sessions/

# Look for the session ID in the list
dsa status <correct_session_id>
```

### Issue: "LangSmith traces not appearing"

1. **Verify API key:**
   ```bash
   echo $OBSERVABILITY__LANGSMITH_API_KEY
   ```

2. **Enable debug logging:**
   ```env
   LOGGING__LEVEL=DEBUG
   ```

3. **Check network connectivity:**
   ```bash
   curl https://api.smith.langchain.com/
   ```

4. **Verify project name:**
   - Go to https://smith.langchain.com/
   - Check project exists and matches config

### Issue: "HITL not triggering approval"

1. **Check confidence calculation:**
   ```python
   # In your agent code, ensure CriticContext.confidence is set
   context = CriticContext(
       passed=False,
       confidence=0.65,  # Must be set!
       ...
   )
   ```

2. **Verify threshold:**
   ```bash
   python <<'EOF'
   from main import load_session_checkpoint
   state = load_session_checkpoint("my_session")
   threshold = state.user_preferences.get("hitl_confidence_threshold", 0.7)
   print(f"HITL Threshold: {threshold}")
   EOF
   ```

3. **Check enable_hitl flag:**
   ```python
   state.user_preferences.get("enable_hitl", True)
   ```

### Issue: "SQLite database locked"

```bash
# Close any other connections to the database
# Check if another process is using it:
lsof | grep dsa_checkpoints.db

# Delete and recreate:
rm data/checkpoints/*.db
dsa run data/sample.csv --session-id fresh_run --sqlite
```

### Issue: "Out of memory with large datasets"

```python
# Set batch size or chunking in config
AUTOML__MAX_TRIALS=20  # Reduce trials
AUTOML__TIMEOUT_SECONDS=1800  # Reduce timeout

# Or use streaming with smaller chunks
dsa run data/large_dataset.csv --session-id streaming_run
```

---

## Next Steps

1. **Read UPGRADES.md** for detailed feature documentation
2. **Explore agent nodes** in `src/agents/*/node.py`
3. **Customize routing** by modifying `src/core/graph.py`
4. **Add custom observability** using `src/core/observability.py` utilities
5. **Deploy to production** with SQLite or PostgreSQL checkpointing

---

## Support & Resources

- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **LangSmith Docs:** https://docs.smith.langchain.com/
- **Typer Docs:** https://typer.tiangolo.com/
- **Pydantic Docs:** https://docs.pydantic.dev/latest/

---

For issues or questions, check the troubleshooting section or review the example usage in `UPGRADES.md`.
