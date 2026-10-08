import os
import sqlite3
import re
from functools import wraps

class SQLiteDictCursor:
    """Cursor wrapper that mimics MySQL DictCursor for SQLite."""
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()

    def execute(self, query, params=None):
        # Convert %s placeholders to ? for SQLite
        sqlite_query = re.sub(r'%s', '?', query)
        if params is None:
            self.cursor.execute(sqlite_query)
        else:
            if isinstance(params, dict):
                self.cursor.execute(sqlite_query, params)
            else:
                self.cursor.execute(sqlite_query, tuple(params))
        return self.cursor.rowcount

    def fetchone(self):
        row = self.cursor.fetchone()
        if row is None:
            return None
        columns = [col[0] for col in self.cursor.description]
        return dict(zip(columns, row))

    def fetchall(self):
        rows = self.cursor.fetchall()
        if not rows:
            return []
        columns = [col[0] for col in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def close(self):
        self.cursor.close()

class SQLiteConnectionWrapper:
    """Connection wrapper mimicking Flask_MySQLdb connection object."""
    def __init__(self, db_path):
        self.db_path = db_path
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def cursor(self):
        return SQLiteDictCursor(self._get_conn())

    def commit(self):
        if self._conn:
            self._conn.commit()

    def rollback(self):
        if self._conn:
            self._conn.rollback()

class DBAdapter:
    def __init__(self, app=None):
        self.use_mysql = False
        self.mysql = None
        self.sqlite_wrapper = None
        self.db_path = None
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        db_dir = os.path.join(app.root_path, 'DB')
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, 'quizapp.db')

        # Try initializing MySQL first if configured
        try:
            from flask_mysqldb import MySQL
            self.mysql = MySQL(app)
            # Test connection
            with app.app_context():
                try:
                    conn = self.mysql.connection
                    if conn:
                        cur = conn.cursor()
                        cur.execute("SELECT 1")
                        cur.close()
                        self.use_mysql = True
                        print("[DBAdapter] Connected to MySQL database successfully.")
                except Exception as e:
                    print(f"[DBAdapter] MySQL connection unavailable ({e}). Falling back to SQLite.")
                    self.use_mysql = False
        except ImportError:
            print("[DBAdapter] Flask_MySQLdb not installed. Using SQLite database adapter.")
            self.use_mysql = False

        if not self.use_mysql:
            self.sqlite_wrapper = SQLiteConnectionWrapper(self.db_path)
            self._init_sqlite_db()

    @property
    def connection(self):
        if self.use_mysql and self.mysql and self.mysql.connection:
            return self.mysql.connection
        return self.sqlite_wrapper

    def _init_sqlite_db(self):
        """Creates tables in SQLite if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Schema statements adapted for SQLite
        tables = [
            """
            CREATE TABLE IF NOT EXISTS users (
                uid INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                register_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_type TEXT NOT NULL,
                user_image TEXT NOT NULL,
                user_login INTEGER NOT NULL DEFAULT 0,
                examcredits INTEGER NOT NULL DEFAULT 7
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS teachers (
                tid INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                test_id TEXT NOT NULL,
                test_type TEXT NOT NULL,
                start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration INTEGER NOT NULL,
                show_ans INTEGER NOT NULL,
                password TEXT NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                neg_marks INTEGER NOT NULL,
                calc INTEGER NOT NULL,
                proctoring_type INTEGER NOT NULL DEFAULT 0,
                uid INTEGER NOT NULL,
                FOREIGN KEY (uid) REFERENCES users (uid)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS questions (
                questions_uid INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                qid TEXT NOT NULL,
                q TEXT NOT NULL,
                a TEXT NOT NULL,
                b TEXT NOT NULL,
                c TEXT NOT NULL,
                d TEXT NOT NULL,
                ans TEXT NOT NULL,
                marks INTEGER NOT NULL,
                uid INTEGER NOT NULL,
                FOREIGN KEY (uid) REFERENCES users (uid)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS longqa (
                longqa_qid INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                qid TEXT NOT NULL,
                q TEXT NOT NULL,
                marks INTEGER DEFAULT NULL,
                uid INTEGER DEFAULT NULL,
                FOREIGN KEY (uid) REFERENCES users (uid)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS longtest (
                longtest_qid INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                test_id TEXT NOT NULL,
                qid INTEGER NOT NULL,
                ans TEXT NOT NULL,
                marks INTEGER NOT NULL,
                uid INTEGER NOT NULL,
                FOREIGN KEY (uid) REFERENCES users (uid)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS practicalqa (
                pracqa_qid INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                qid TEXT NOT NULL,
                q TEXT NOT NULL,
                compiler INTEGER NOT NULL,
                marks INTEGER NOT NULL,
                uid INTEGER NOT NULL,
                FOREIGN KEY (uid) REFERENCES users (uid)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS practicaltest (
                pid INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                test_id TEXT NOT NULL,
                qid TEXT NOT NULL,
                code TEXT,
                input TEXT,
                executed TEXT DEFAULT NULL,
                marks INTEGER NOT NULL,
                uid INTEGER NOT NULL,
                FOREIGN KEY (uid) REFERENCES users (uid)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS proctoring_log (
                pid INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                name TEXT NOT NULL,
                test_id TEXT NOT NULL,
                voice_db INTEGER DEFAULT 0,
                img_log TEXT NOT NULL,
                user_movements_updown INTEGER NOT NULL,
                user_movements_lr INTEGER NOT NULL,
                user_movements_eyes INTEGER NOT NULL,
                phone_detection INTEGER NOT NULL,
                person_status INTEGER NOT NULL,
                log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                uid INTEGER NOT NULL,
                FOREIGN KEY (uid) REFERENCES users (uid)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS students (
                sid INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                test_id TEXT NOT NULL,
                qid TEXT DEFAULT NULL,
                ans TEXT,
                uid INTEGER NOT NULL,
                FOREIGN KEY (uid) REFERENCES users (uid)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS studenttestinfo (
                stiid INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                test_id TEXT NOT NULL,
                time_left TEXT NOT NULL,
                completed INTEGER DEFAULT 0,
                uid INTEGER NOT NULL,
                FOREIGN KEY (uid) REFERENCES users (uid)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS window_estimation_log (
                wid INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                test_id TEXT NOT NULL,
                name TEXT NOT NULL,
                window_event INTEGER NOT NULL,
                transaction_log TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                uid INTEGER NOT NULL,
                FOREIGN KEY (uid) REFERENCES users (uid)
            );
            """
        ]

        for table_sql in tables:
            cur.execute(table_sql)
        
        conn.commit()
        conn.close()
        print("[DBAdapter] SQLite database initialized at:", self.db_path)
