# DSA Architecture Diagrams

## 1. Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Typer CLI                                                    │   │
│  │  $ dsa run --dataset data.csv --session-id exp1              │   │
│  │  $ dsa resume exp1 --dataset data2.csv                       │   │
│  │  $ dsa list-sessions / status exp1                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Session Management Layer                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Session ID: UUID                                             │   │
│  │  Thread ID: thread_xxxxx                                      │   │
│  │  Storage: ./data/sessions/<session-id>.json                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Execution & State Layer                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  DSAState (Pydantic Model)                                 │     │
│  │  ├─ session_id, thread_id                                  │     │
│  │  ├─ pending_human_approval, approval_request               │     │
│  │  ├─ logs, messages, errors                                 │     │
│  │  ├─ agent contexts (planner, critic, trainer, etc.)        │     │
│  │  └─ metadata (timestamps, execution info)                  │     │
│  │                                                             │     │
│  │  Methods:                                                  │     │
│  │  ├─ request_human_approval(msg)                            │     │
│  │  ├─ approve_and_continue()                                 │     │
│  │  └─ model_dump_for_checkpoint()                            │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph Execution Layer                        │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Compiled StateGraph (with checkpointing & interrupts)     │     │
│  │  ├─ Checkpointer: Memory/SQLite/Postgres                   │     │
│  │  ├─ Interrupts: [critic] (for HITL)                        │     │
│  │  └─ Thread ID config for persistence                       │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Agent Execution Layer                          │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  [Memory] → [Quality] → [Supervisor]                       │     │
│  │                              ↓                              │     │
│  │  [Planner] → [Explorer] → [Trainer] → [Critic] → HITL?     │     │
│  │                                            ↓                │     │
│  │                        [Memory Store]      |                │     │
│  │                            ↓               |                │     │
│  │                           END     [Human Approval]          │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   Observability & Tracing Layer                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  LangSmith Integration                                     │     │
│  │  ├─ Agent execution traces                                 │     │
│  │  ├─ Decision point traces                                  │     │
│  │  ├─ Custom event logging                                   │     │
│  │  └─ Metadata annotations                                   │     │
│  │                                                             │     │
│  │  Configuration:                                            │     │
│  │  ├─ OBSERVABILITY__LANGSMITH_ENABLED=true                  │     │
│  │  ├─ OBSERVABILITY__LANGSMITH_API_KEY=ls_...                │     │
│  │  └─ OBSERVABILITY__LANGSMITH_PROJECT=dsa-agent             │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Persistence & Storage Layer                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Session Metadata:   ./data/sessions/<id>.json             │     │
│  │  Checkpoints (SQLite): ./data/checkpoints/<id>.db          │     │
│  │  Logs:               ./artifacts/logs/<id>.log             │     │
│  │  Models:             ./artifacts/models/<id>/              │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Human-in-the-Loop (HITL) Decision Flow

```
                    ┌─────────────────┐
                    │   Trainer Node  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Critic Node   │
                    │  (Interrupted)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Evaluation     │
                    │  ├─ Passed?     │
                    │  ├─ Confidence? │
                    │  └─ Metrics     │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
        Result1          Result2         Result3
    (High Pass)      (Low Pass/Fail)    (High Fail)
        │                 │                 │
        ▼                 ▼                 ▼
    ╔═══════╗      ╔═══════════════╗   ╔═════════╗
    ║Continue║     ║Request Human  ║   ║Auto-Retry║
    ║to Store║     ║Approval (HITL)║   ║(Backtrack)║
    ╚═══════╝      ╚═══════════════╝   ╚═════════╝
        │                 │                 │
        │          ┌──────┴──────┐          │
        │          │             │          │
        │          ▼             ▼          │
        │       Approve      Reject/Modify   │
        │          │             │          │
        │          ▼             ▼          │
        │      Continue       Feedback       │
        │                    (Retry)         │
        │          │             │          │
        └─────────▶│             │◀────────┘
                   │             │
                   ▼             ▼
            ┌─────────────────────────┐
            │  Route to Supervisor    │
            │  (Next Step Decided)    │
            └─────────────────────────┘
```

---

## 3. Session Lifecycle

```
START
  │
  ├─ [CLI Input] dsa run dataset.csv [--session-id xyz]
  │
  ├─ Create DSAState() or Load existing
  │    ├─ Generate session_id (UUID) if not provided
  │    ├─ Generate thread_id (thread_xxxxx)
  │    └─ Set user_preferences (enable_hitl, thresholds, etc.)
  │
  ├─ Initialize Observability
  │    └─ Configure LangSmith if enabled
  │
  ├─ Build LangGraph
  │    ├─ Select checkpointer (Memory/SQLite/Postgres)
  │    ├─ Set interrupt_before=["critic"] if HITL enabled
  │    └─ Compile graph
  │
  ├─ Stream Graph Execution
  │    │
  │    ├─ Memory → Quality → Supervisor
  │    │
  │    ├─ (Routing Loop)
  │    │   ├─ Planner → Explorer → Trainer → Critic
  │    │   │                          │
  │    │   │                   (Check HITL condition)
  │    │   │                          │
  │    │   │        ┌─────────────────┼──────────────┐
  │    │   │        │                 │              │
  │    │   │   (High Pass)      (HITL Triggered)  (Retry)
  │    │   │        │                 │              │
  │    │   │   Continue            PAUSE              │
  │    │   │   (Log event)         (Wait)          Loop back
  │    │   │        │                 │              │
  │    │   │        ▼                 ▼              │
  │    │   └────────→ Supervisor ←────────────────┘
  │    │
  │    └─ Memory Store → END
  │
  ├─ Save Checkpoint
  │    ├─ JSON: ./data/sessions/{session-id}.json
  │    ├─ SQLite: ./data/checkpoints/{session-id}.db (optional)
  │    └─ Logs: ./artifacts/logs/{session-id}.log
  │
  └─ Return Final State
       │
       ├─ Session ID: preserved for resume
       ├─ Thread ID: preserved for continuity
       ├─ Status: completed/failed/awaiting_approval
       └─ All execution data: saved and retrievable

RESUME (dsa resume session-id)
  │
  ├─ Load session checkpoint from JSON
  │    └─ Restore entire DSAState
  │
  ├─ Check if pending_human_approval
  │    │
  │    ├─ YES: Handle approval logic
  │    │       ├─ state.approve_and_continue()
  │    │       └─ Resume graph with updated state
  │    │
  │    └─ NO: Continue from last checkpoint
  │
  └─ [Same as stream/save cycle above]
```

---

## 4. Confidence-Based Routing in Critic

```
    Critic Evaluation Complete
            │
            ▼
    ┌───────────────────┐
    │  Calculate Score  │
    │  confidence =     │
    │  (acc+f1+         │
    │   prec+rec)/4     │
    └───────┬───────────┘
            │
            ▼
    ┌───────────────────────┐
    │  Get Configuration    │
    │  ├─ enable_hitl      │ (from user_preferences)
    │  └─ threshold: 0.7   │
    └───────┬───────────────┘
            │
            ▼
    ╔══════════════════════════════════════════════════════╗
    ║  Passed && Confidence > Threshold                    ║
    ║  → Continue to Memory Store (No HITL needed)         ║
    ╚══════════════════════════════════════════════════════╝
            │
            ▼
    ╔══════════════════════════════════════════════════════╗
    ║  !Passed && Confidence >= Threshold                 ║
    ║  → High confidence failure                           ║
    ║  → state.metadata["next_agent"] = "retry"            ║
    ║  → Route back to Supervisor (Auto-retry)             ║
    ╚══════════════════════════════════════════════════════╝
            │
            ▼
    ╔══════════════════════════════════════════════════════╗
    ║  !Passed && Confidence < Threshold && HITL enabled   ║
    ║  → Low confidence failure (ambiguous case)           ║
    ║  → state.request_human_approval(...)                 ║
    ║  → state.pending_human_approval = True               ║
    ║  → Route back to Supervisor (PAUSES HERE)            ║
    ║  → Workflow awaits human decision                    ║
    ╚══════════════════════════════════════════════════════╝
            │
            ▼
    ╔══════════════════════════════════════════════════════╗
    ║  Human Review Needed                                ║
    ║  $ dsa status {session-id}                          ║
    ║  → Shows: "Pending Human Approval"                  ║
    ║  $ dsa resume {session-id}                          ║
    ║  → Approves and continues                            ║
    ║  → OR modify state and retry                         ║
    ╚══════════════════════════════════════════════════════╝
```

---

## 5. Observability Integration

```
┌────────────────────────────────────────────────────────────────┐
│                    LangSmith Configuration                     │
│  .env or config.yaml:                                          │
│  ├─ LANGSMITH_ENABLED=true                                    │
│  ├─ LANGSMITH_API_KEY=ls_xxxxx                                 │
│  └─ LANGSMITH_PROJECT=dsa-agent                                │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                  init_observability()                          │
│  Called once at startup to configure tracing                   │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
            ┌───────────────┬───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  trace_      │  │  trace_      │  │  log_trace   │
    │  agent_      │  │  decision_   │  │  _event()    │
    │  execution() │  │  point()     │  │              │
    └──────────────┘  └──────────────┘  └──────────────┘
            │               │                   │
            │               │                   │
    Context Manager  Context Manager    Simple Function
    ├─ Start Time    ├─ Decision Name    ├─ Event Type
    ├─ Agent Name    ├─ Input Data       ├─ Metadata
    ├─ Session ID    ├─ Reasoning        └─ Optional Details
    ├─ Thread ID     └─ Output Decision
    └─ Metadata
            │
            ▼
    ┌──────────────────────────────────────┐
    │   Automatic Capture (LangSmith)      │
    │   ├─ Execution timing                │
    │   ├─ Input/Output states              │
    │   ├─ Error handling                   │
    │   └─ Metadata annotations             │
    └──────────────────────────────────────┘
            │
            ▼
    https://smith.langchain.com/projects/dsa-agent
    │
    ├─ Project Dashboard
    ├─ Trace Timeline
    ├─ Performance Metrics
    ├─ Error Analysis
    └─ Decision Flow Visualization
```

---

## 6. Data Flow Diagram

```
                    External Data Source
                            │
                            ▼
                    ┌─────────────────┐
                    │  Dataset Input  │
                    │  (CSV/Parquet)  │
                    └────────┬────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │         CLI Session Creation           │
        │  dsa run data.csv --session-id exp1    │
        └────────────┬─────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │      DSAState Initialization           │
        │  ├─ Dataset reference                 │
        │  ├─ Session ID (UUID)                 │
        │  ├─ Thread ID (thread_xxxxx)          │
        │  └─ Empty contexts & logs             │
        └────────────┬─────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │        Agent Processing Loop           │
        │  (Supervisor orchestrates flow)        │
        └────────────┬─────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
    ┌─────────┐         ┌──────────────┐
    │ Planner │ ─────→  │ Explorer     │
    └─────────┘         └──────┬───────┘
                                │
                                ▼
                        ┌──────────────┐
                        │ Trainer      │
                        │ (ML Models)  │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │ Critic           │
                        │ (Evaluation)     │
                        └──────┬────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
    (Pass)              (HITL Needed)          (Retry)
        │                      │                      │
        ▼                      ▼                      ▼
    Continue            Request Approval      Back to Supervisor
        │                      │                      │
        │              (Human Reviews)                │
        │                      │                      │
        │          ┌───────────┴─────────┐            │
        │          ▼                     ▼            │
        │     (Approve)             (Modify)          │
        │          │                     │            │
        └──────────┴─────────────────────┘
                    │
                    ▼
        ┌────────────────────────────────────────┐
        │  Memory Store (Save Results)           │
        └────────────┬─────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │   Save Checkpoint                      │
        │   ├─ JSON: ./data/sessions/            │
        │   ├─ SQLite: ./data/checkpoints/       │
        │   └─ Logs: ./artifacts/logs/           │
        └────────────┬─────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │   Return Final State                   │
        │   ├─ Results & Artifacts               │
        │   ├─ Execution Logs                    │
        │   └─ Session Metadata                  │
        └────────────────────────────────────────┘
```

---

## 7. File Dependencies

```
main.py (Entry Point)
    │
    ├─→ src/core/config.py
    │       └─→ Loads settings from .env/config.yaml
    │
    ├─→ src/core/graph.py
    │       ├─→ src/core/state.py (DSAState model)
    │       ├─→ src/core/observability.py (Tracing)
    │       └─→ src/agents/
    │           ├─ supervisor/node.py
    │           ├─ planner/node.py
    │           ├─ explorer/node.py
    │           ├─ trainer/node.py
    │           ├─ critic/node.py
    │           ├─ quality/node.py
    │           └─ memory/node.py
    │
    ├─→ src/core/observability.py
    │       ├─→ src/core/config.py (Settings)
    │       └─→ langsmith (Optional: LangSmith SDK)
    │
    ├─→ src/utils/logger.py
    │       └─→ Provides logging utility
    │
    └─→ External Dependencies
        ├─ typer (CLI framework)
        ├─ rich (Console output)
        ├─ pydantic (Data validation)
        ├─ langgraph (Graph orchestration)
        ├─ langchain (LLM utilities)
        └─ langsmith (Optional: Observability)
```

These diagrams provide a comprehensive visual reference for the DSA architecture, execution flow, HITL decision logic, session lifecycle, and data flow throughout the system.
