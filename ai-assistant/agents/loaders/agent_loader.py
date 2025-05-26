import psycopg2
import os
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.expanduser("~/ai-assistant/config/.env"))

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT")
    )

def load_user_agents(user_id: int):
    """
    Fetch all AI agents (personalities) defined by a specific user.
    Returns a list of dictionaries with agent name, description, and config.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT name, description, personality_config
        FROM user_personalities
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (user_id,))

    agents = []
    for row in cur.fetchall():
        name, desc, config_json = row
        config = config_json if isinstance(config_json, dict) else json.loads(config_json)
        agents.append({
            "name": name,
            "description": desc,
            "config": config
        })

    cur.close()
    conn.close()
    return agents

if __name__ == "__main__":
    import pprint
    user_id = 1  # for test/demo
    pprint.pprint(load_user_agents(user_id))