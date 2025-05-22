#!/bin/bash

BASEDIR="./ai-assistant"

echo "📁 Creating project structure at $BASEDIR"

# Core engine logic
mkdir -p "$BASEDIR/core"
touch "$BASEDIR/core/engine.py"
touch "$BASEDIR/core/context_builder.py"
touch "$BASEDIR/core/router.py"

# Memory modules
mkdir -p "$BASEDIR/memory"
touch "$BASEDIR/memory/buffer.py"
touch "$BASEDIR/memory/vector_store.py"
touch "$BASEDIR/memory/fact_store.py"
touch "$BASEDIR/memory/topic_graph.py"
touch "$BASEDIR/memory/classifier.py"

# Agents
mkdir -p "$BASEDIR/agents"
touch "$BASEDIR/agents/personality_config.json"
touch "$BASEDIR/agents/loader.py"

# Tools
mkdir -p "$BASEDIR/tools"
touch "$BASEDIR/tools/image_gen.py"
touch "$BASEDIR/tools/sandbox_env.py"
touch "$BASEDIR/tools/web_research.py"

# Interface
mkdir -p "$BASEDIR/interface"
touch "$BASEDIR/interface/cli_chat.py"
mkdir -p "$BASEDIR/interface/web"
mkdir -p "$BASEDIR/interface/discord"
mkdir -p "$BASEDIR/interface/api"

# Config
mkdir -p "$BASEDIR/config"
touch "$BASEDIR/config/.env"

# Scripts
mkdir -p "$BASEDIR/scripts"
touch "$BASEDIR/scripts/init_postgres.py"
touch "$BASEDIR/scripts/init_neo4j.py"
touch "$BASEDIR/scripts/init_qdrant.py"
touch "$BASEDIR/scripts/init_structure.sh"

# Data
mkdir -p "$BASEDIR/data/echo_corpus"

echo "✅ Folder structure created successfully."
