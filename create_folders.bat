@echo off
SET BASEDIR=%~dp0ai-assistant

REM Core engine logic
mkdir "%BASEDIR%\core"
type nul > "%BASEDIR%\core\engine.py"
type nul > "%BASEDIR%\core\context_builder.py"
type nul > "%BASEDIR%\core\router.py"

REM Memory modules
mkdir "%BASEDIR%\memory"
type nul > "%BASEDIR%\memory\buffer.py"
type nul > "%BASEDIR%\memory\vector_store.py"
type nul > "%BASEDIR%\memory\fact_store.py"
type nul > "%BASEDIR%\memory\topic_graph.py"
type nul > "%BASEDIR%\memory\classifier.py"

REM Agents
mkdir "%BASEDIR%\agents"
type nul > "%BASEDIR%\agents\personality_config.json"
type nul > "%BASEDIR%\agents\loader.py"

REM Tools
mkdir "%BASEDIR%\tools"
type nul > "%BASEDIR%\tools\image_gen.py"
type nul > "%BASEDIR%\tools\sandbox_env.py"
type nul > "%BASEDIR%\tools\web_research.py"

REM Interfaces
mkdir "%BASEDIR%\interface"
type nul > "%BASEDIR%\interface\cli_chat.py"
mkdir "%BASEDIR%\interface\web"
mkdir "%BASEDIR%\interface\discord"
mkdir "%BASEDIR%\interface\api"

REM Config
mkdir "%BASEDIR%\config"
type nul > "%BASEDIR%\config\.env"

REM Scripts
mkdir "%BASEDIR%\scripts"
type nul > "%BASEDIR%\scripts\init_postgres.py"
type nul > "%BASEDIR%\scripts\init_neo4j.py"
type nul > "%BASEDIR%\scripts\init_qdrant.py"
type nul > "%BASEDIR%\scripts\init_structure.sh"

REM Data
mkdir "%BASEDIR%\data"
mkdir "%BASEDIR%\data\echo_corpus"

echo Folder structure created under %BASEDIR%
pause
