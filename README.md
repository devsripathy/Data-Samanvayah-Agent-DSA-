# 🧠 Data Samanvayah Agent (DSA)

**An Enterprise-Grade, Multi-Agent AutoML System powered by LangGraph.**

DSA (Data Samanvayah - "Data Coordination" in Sanskrit) is not just an AutoML tool; it is a stateful, multi-agent orchestration system designed to mimic a team of data scientists. It utilizes episodic memory to learn from past executions, dynamically plans preprocessing steps, and employs a critic-driven retry loop to ensure model quality.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](./Dockerfile)
[![CI Status](https://github.com/yourusername/data-samanvayah-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/data-samanvayah-agent/actions)

---

## 🏗️ Architecture & Agent Orchestration

DSA uses a **StateGraph** where specialized agents pass a unified `DSAState` object. The `Supervisor` acts as the central router, while the `Critic` enables dynamic, LLM-evaluated retry loops.

```mermaid
graph TD
    A[User Input / CLI] --> B(Memory Agent)
    B --> C(Quality Agent)
    C --> D{Supervisor}
    D -->|Initial Run| E(Planner Agent)
    E --> F(Explorer Agent)
    F --> G(Trainer Agent)
    G --> H(Critic Agent)
    H -->|Pass Threshold| I[Output Artifacts & Model]
    H -->|Fail & Retries Left| D
    H -->|Fail & Max Retries| J[End with Error Logs]
    
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
