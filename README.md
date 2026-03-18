# CYBEL

An **open-source agentic workflow engine** and execution layer for orchestrating autonomous AI systems **locally**.

Cybel provides a modular runtime that allows AI agents to reason, retrieve knowledge, interact with tools, and coordinate workflows across multiple model providers.

The goal is simple:

**Give developers a flexible system for building autonomous AI pipelines that run locally, integrate external data, and remain fully programmable.**

---

# Key Features

### Agentic Execution Layer

* Multi-agent orchestration
* Structured reasoning pipelines
* Tool routing based on detected intent

### Retrieval-Augmented Generation (RAG)

* FAISS vector search
* Local embedding pipelines
* JSON document ingestion
* Persistent vector index caching

### Multi-Model Support

Supports switching between multiple inference backends:

* Groq (OpenAI, Qwen, and more!)
* xAI Grok
* Local Transformers
* Verso pipeline *(experimental)*

### Data Ingestion

Cybel can ingest knowledge from multiple sources:

* JSON knowledge bases
* Uploaded documents
* Conversation logs
* Google Sheets ingestion
* Local structured datasets

### Tool Execution

Agents can automatically route tasks to specialized generators:

* Text generation
* Image generation
* 3D model generation
* Voice synthesis
* Video pipelines

---

# Architecture Overview

```
User Input
     │
Intent Detection
     │
Generator Router
     │
 ┌───────────────┐
 │  Tool Models  │
 │  Image / 3D   │
 │  Video / TTS  │
 └───────────────┘
     │
Text Pipeline
     │
Embedding Model
     │
FAISS Vector Retrieval
     │
Context Builder
     │
LLM Generation
     │
Response
```

---

# Google Sheets Ingestion

Cybel can ingest **structured spreadsheet data directly from Google Sheets**.

This enables workflows where agents reason over live datasets without manual exports.

Workflow:

1. Authenticate with Google Drive
2. Select spreadsheets through a dialog interface
3. Convert rows → structured documents
4. Embed and add to FAISS vector index
5. Immediately available for RAG queries

---

# Example Use Cases

* Autonomous research assistants
* Data-aware AI copilots
* Local AI knowledge bases
* Multi-modal generation pipelines
* AI-powered document analysis

---

# Installation

```
git clone https://github.com/Team-Ferrous/automat-llm
cd automat-llm
npm install
```

Run the application:

```
npm start
```

---

# Current Status

⚠️ **Active Development**

Some components are still being refined:

* Verso inference pipeline not fully stable
* Extensive testing still required
* Additional model integrations planned

Despite this, the system is already capable of running **end-to-end agentic workflows locally**.

---

# Roadmap

* Improved workflow orchestration
* Expanded tool execution layer
* Multi-agent collaboration
* Plugin architecture
* Dataset connectors
* Improved UI
* Performance optimizations

---

# Contributing

Contributions are welcome.

If you'd like to help:

* Test the pipelines
* Report bugs
* Improve integrations
* Add new tools or models

Open an issue or submit a pull request.

---

# Vision

Automat-LLM aims to become a **general-purpose open execution layer for autonomous AI systems**.

Not just a chatbot.

A programmable runtime for building **AI-driven software systems.**

---

# License

Automat-LLM is released under the **PolyForm Noncommercial License 1.0.0**.

This means:

You are free to:

• Use the software for personal projects
• Use it for research and experimentation
• Contribute improvements to the project
• Study and modify the code

However:

Commercial use of Automat-LLM **requires a commercial license from the authors.**

If you would like to use Automat-LLM in a commercial product, company, or paid service, please contact:

**Team Ferrous**
Team Lead: mileslitteral@sasorizerolabs.com
General Team Inquiry: sasorizerolabs@gmail.com

Commercial licensing inquiries can be opened through GitHub issues or direct contact with the maintainers.

See the full license text in the `LICENSE` file.

