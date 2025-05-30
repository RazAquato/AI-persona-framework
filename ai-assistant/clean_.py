@classmethod
def setUpClass(cls):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ENV_PATH = os.path.join(BASE_DIR, "..", "config", ".env")
    load_dotenv(dotenv_path=os.path.abspath(ENV_PATH))

    cls.conn = psycopg2.connect(
        dbname=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT")
    )
    cls.cur = cls.conn.cursor()
    cls.cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id;", ("TestUser",))
    cls.test_user_id = cls.cur.fetchone()[0]
    cls.conn.commit()

    # ✅ Delete all facts for this user without relevance_score
    cls.cur.execute("""
        DELETE FROM facts
        WHERE user_id = %s AND relevance_score IS NULL;
    """, (cls.test_user_id,))
    cls.conn.commit()

