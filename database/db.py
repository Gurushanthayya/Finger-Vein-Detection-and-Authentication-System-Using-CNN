"""
Database module supporting SQLite and MySQL fallback.
Tracks users, finger vein templates, logs, stats, and evaluation runs.
"""
import os
import uuid
import sqlite3
import numpy as np
from config import Config

try:
    import mysql.connector
    from mysql.connector import pooling
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False


class SQLiteCursorWrapper:
    def __init__(self, sqlite_cursor):
        self.cursor = sqlite_cursor

    def execute(self, query, params=None):
        query_upper = query.strip().upper()
        
        # Translate MySQL specific check commands
        if "SET FOREIGN_KEY_CHECKS" in query_upper:
            if "0" in query_upper:
                query = "PRAGMA foreign_keys = OFF;"
            else:
                query = "PRAGMA foreign_keys = ON;"
            params = None
        
        # Translate MySQL TRUNCATE to SQLite DELETE
        elif query_upper.startswith("TRUNCATE TABLE"):
            table_name = query.split()[-1].strip(";")
            self.cursor.execute(f"DELETE FROM {table_name};")
            try:
                self.cursor.execute("DELETE FROM sqlite_sequence WHERE name = ?;", (table_name,))
            except Exception:
                pass
            return self
            
        # Placeholders: convert %s to ?
        if "%s" in query:
            query = query.replace("%s", "?")
            
        if params is not None:
            if not isinstance(params, (tuple, list)):
                params = (params,)
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        return self

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()


class SQLiteConnectionWrapper:
    def __init__(self, sqlite_path):
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def cursor(self):
        return SQLiteCursorWrapper(self.conn.cursor())

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()


class DatabaseManager:
    def __init__(self, db_name=None):
        self.use_sqlite = True
        
        # Override DB names if testing
        if db_name:
            self.sqlite_path = os.path.join(Config.BASE_DIR, 'database', f"{db_name}.sqlite")
            self.mysql_db = db_name
        else:
            self.sqlite_path = Config.SQLITE_PATH
            self.mysql_db = Config.MYSQL_DATABASE
            
        # Ensure database folder exists
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        
        if Config.DB_TYPE == 'mysql':
            if HAS_MYSQL:
                try:
                    # Connect without database first to ensure the database exists
                    temp_conn = mysql.connector.connect(
                        host=Config.MYSQL_HOST,
                        port=Config.MYSQL_PORT,
                        user=Config.MYSQL_USER,
                        password=Config.MYSQL_PASSWORD
                    )
                    temp_cursor = temp_conn.cursor()
                    temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.mysql_db}")
                    temp_cursor.close()
                    temp_conn.close()
                    
                    self.pool = mysql.connector.pooling.MySQLConnectionPool(
                        pool_name="veinid_pool", 
                        pool_size=5, 
                        pool_reset_session=True,
                        host=Config.MYSQL_HOST,
                        port=Config.MYSQL_PORT,
                        user=Config.MYSQL_USER,
                        password=Config.MYSQL_PASSWORD,
                        database=self.mysql_db,
                    )
                    self.use_sqlite = False
                    print(f"[Database] Connected to MySQL database: {self.mysql_db}")
                    self._init_mysql()
                except Exception as e:
                    print(f"[Database] MySQL connect failed ({e}). Falling back to SQLite.")
                    self.use_sqlite = True
            else:
                print("[Database] mysql-connector-python not installed. Falling back to SQLite.")
                self.use_sqlite = True

        if self.use_sqlite:
            self._init_sqlite()

    def get_connection(self):
        if self.use_sqlite:
            return SQLiteConnectionWrapper(self.sqlite_path)
        else:
            return self.pool.get_connection()

    def _init_mysql(self):
        conn = self.pool.get_connection()
        c = conn.cursor()
        try:
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    uid         VARCHAR(8)   PRIMARY KEY,
                    name        VARCHAR(255) NOT NULL,
                    user_id     VARCHAR(255) UNIQUE NOT NULL,
                    registered_at DATETIME   DEFAULT CURRENT_TIMESTAMP,
                    is_active   BOOLEAN      DEFAULT TRUE
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS vein_templates (
                    id           INT AUTO_INCREMENT PRIMARY KEY,
                    user_uid     VARCHAR(8)   NOT NULL,
                    feature_blob LONGBLOB     NOT NULL,
                    quality      FLOAT        DEFAULT 0.0,
                    enrolled_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_uid) REFERENCES users(uid) ON DELETE CASCADE
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS auth_logs (
                    id              INT AUTO_INCREMENT PRIMARY KEY,
                    user_uid        VARCHAR(8)   NULL,
                    matched_name    VARCHAR(255) NULL,
                    timestamp       DATETIME     DEFAULT CURRENT_TIMESTAMP,
                    confidence      FLOAT        NOT NULL,
                    result          ENUM('accept', 'reject') NOT NULL,
                    processing_ms   FLOAT        DEFAULT 0.0,
                    ip_address      VARCHAR(45)  NULL
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS user_stats (
                    user_uid      VARCHAR(8) PRIMARY KEY,
                    auth_count    INT     DEFAULT 0,
                    last_auth     DATETIME NULL,
                    avg_confidence FLOAT  DEFAULT 0.0,
                    FOREIGN KEY (user_uid) REFERENCES users(uid) ON DELETE CASCADE
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
                    accuracy    FLOAT,
                    precision_val FLOAT,
                    recall      FLOAT,
                    f1_score    FLOAT,
                    far         FLOAT,
                    frr         FLOAT
                );
            """)
            conn.commit()
            print("[Database] MySQL schema verified.")
        except Exception as e:
            print(f"[Database] Error creating MySQL tables: {e}")
        finally:
            c.close()
            conn.close()

    def _init_sqlite(self):
        conn = sqlite3.connect(self.sqlite_path)
        c = conn.cursor()
        c.execute("PRAGMA foreign_keys = ON;")
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                uid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                user_id TEXT UNIQUE NOT NULL,
                registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS vein_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_uid TEXT NOT NULL,
                feature_blob BLOB NOT NULL,
                enrolled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_uid) REFERENCES users(uid) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_uid TEXT NULL,
                matched_name TEXT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL NOT NULL,
                result TEXT NOT NULL,
                ip_address TEXT NULL
            );
            CREATE TABLE IF NOT EXISTS user_stats (
                user_uid TEXT PRIMARY KEY,
                auth_count INTEGER DEFAULT 0,
                last_auth DATETIME NULL,
                avg_confidence REAL DEFAULT 0.0,
                FOREIGN KEY (user_uid) REFERENCES users(uid) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                accuracy REAL,
                precision_val REAL,
                recall REAL,
                f1_score REAL,
                far REAL,
                frr REAL
            );
        """)
        conn.commit()
        conn.close()
        print(f"[Database] SQLite DB ready: {self.sqlite_path}")

    def _fmt_dt(self, dt):
        if not dt:
            return None
        if isinstance(dt, str):
            return dt.replace(' ', 'T')
        if hasattr(dt, 'isoformat'):
            return dt.isoformat()
        return str(dt)

    def _cosine(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ── REGISTER USER ────────────────────────────────────────────────────────
    def register(self, name, user_id, features_list):
        uid = str(uuid.uuid4())[:8]
        if self.use_sqlite:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                c = conn.cursor()
                c.execute("INSERT INTO users (uid, name, user_id) VALUES (?, ?, ?)", (uid, name, user_id))
                c.execute("INSERT INTO user_stats (user_uid) VALUES (?)", (uid,))
                for feat in features_list:
                    c.execute("INSERT INTO vein_templates (user_uid, feature_blob) VALUES (?, ?)",
                              (uid, np.array(feat, dtype=np.float32).tobytes()))
                conn.commit()
        else:
            conn = self.pool.get_connection()
            c = conn.cursor()
            try:
                c.execute("INSERT INTO users (uid, name, user_id) VALUES (%s, %s, %s)", (uid, name, user_id))
                c.execute("INSERT INTO user_stats (user_uid) VALUES (%s)", (uid,))
                for feat in features_list:
                    c.execute("INSERT INTO vein_templates (user_uid, feature_blob) VALUES (%s, %s)",
                              (uid, np.array(feat, dtype=np.float32).tobytes()))
                conn.commit()
            finally:
                c.close()
                conn.close()
        return {'uid': uid, 'name': name, 'user_id': user_id}

    # ── AUTHENTICATE USER ────────────────────────────────────────────────────
    def authenticate(self, query_features, threshold=None, ip_address=None):
        if threshold is None:
            threshold = Config.THRESHOLD
        query = np.array(query_features, dtype=np.float32)
        best_uid, best_name, best_score = None, None, -1.0

        if self.use_sqlite:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("""
                    SELECT t.user_uid, t.feature_blob, u.name
                    FROM vein_templates t JOIN users u ON t.user_uid = u.uid
                    WHERE u.is_active = 1
                """)
                for row in c.fetchall():
                    vec = np.frombuffer(row['feature_blob'], dtype=np.float32)
                    if vec.shape != query.shape:
                        continue
                    score = self._cosine(query, vec)
                    if score > best_score:
                        best_score, best_uid, best_name = score, row['user_uid'], row['name']

                match = best_score >= threshold
                result_str = 'accept' if match else 'reject'
                uid_log = best_uid if match else None
                name_log = best_name if match else None

                c.execute("""INSERT INTO auth_logs (user_uid, matched_name, confidence, result, ip_address)
                             VALUES (?, ?, ?, ?, ?)""",
                          (uid_log, name_log, float(best_score), result_str, ip_address))
                if match and best_uid:
                    c.execute("SELECT auth_count FROM user_stats WHERE user_uid = ?", (best_uid,))
                    row = c.fetchone()
                    n = row[0] if row else 0
                    c.execute("""UPDATE user_stats SET auth_count = auth_count + 1,
                                 last_auth = CURRENT_TIMESTAMP,
                                 avg_confidence = ((avg_confidence * ?) + ?) / (? + 1)
                                 WHERE user_uid = ?""",
                              (n, float(best_score), n, best_uid))
                conn.commit()
        else:
            conn = self.pool.get_connection()
            c = conn.cursor(dictionary=True)
            try:
                c.execute("""SELECT t.user_uid, t.feature_blob, u.name
                             FROM vein_templates t JOIN users u ON t.user_uid = u.uid
                             WHERE u.is_active = TRUE""")
                for row in c.fetchall():
                    vec = np.frombuffer(row['feature_blob'], dtype=np.float32)
                    if vec.shape != query.shape:
                        continue
                    score = self._cosine(query, vec)
                    if score > best_score:
                        best_score, best_uid, best_name = score, row['user_uid'], row['name']

                match = best_score >= threshold
                result_str = 'accept' if match else 'reject'
                uid_log = best_uid if match else None
                name_log = best_name if match else None

                c.execute("""INSERT INTO auth_logs (user_uid, matched_name, confidence, result, ip_address)
                             VALUES (%s, %s, %s, %s, %s)""",
                          (uid_log, name_log, float(best_score), result_str, ip_address))
                if match and best_uid:
                    c.execute("SELECT auth_count FROM user_stats WHERE user_uid = %s", (best_uid,))
                    row = c.fetchone()
                    n = row['auth_count'] if row else 0
                    c.execute("""UPDATE user_stats SET auth_count = auth_count + 1,
                                 last_auth = CURRENT_TIMESTAMP,
                                 avg_confidence = ((avg_confidence * %s) + %s) / (%s + 1)
                                 WHERE user_uid = %s""",
                              (n, float(best_score), n, best_uid))
                conn.commit()
            finally:
                c.close()
                conn.close()

        return {
            'match': match,
            'score': float(best_score),
            'matched_user': best_name if match else None,
            'matched_uid': best_uid if match else None,
        }

    # ── UTILITIES ────────────────────────────────────────────────────────────
    def get_all(self):
        q = """
            SELECT u.uid, u.name, u.user_id, u.registered_at,
                   s.auth_count, s.last_auth, s.avg_confidence,
                   (SELECT COUNT(*) FROM vein_templates t WHERE t.user_uid = u.uid) as template_count
            FROM users u LEFT JOIN user_stats s ON u.uid = s.user_uid
            WHERE u.is_active = 1 ORDER BY u.registered_at DESC
        """
        if self.use_sqlite:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = [dict(r) for r in conn.execute(q).fetchall()]
                for r in rows:
                    r['registered_at'] = self._fmt_dt(r['registered_at'])
                    r['last_auth'] = self._fmt_dt(r['last_auth'])
                return rows
        else:
            conn = self.pool.get_connection()
            c = conn.cursor(dictionary=True)
            try:
                c.execute(q.replace('= 1', '= TRUE'))
                rows = c.fetchall()
                for r in rows:
                    r['registered_at'] = self._fmt_dt(r.get('registered_at'))
                    r['last_auth'] = self._fmt_dt(r.get('last_auth'))
                return rows
            finally:
                c.close()
                conn.close()

    def delete_user(self, uid):
        if self.use_sqlite:
            with sqlite3.connect(self.sqlite_path) as conn:
                c = conn.execute("UPDATE users SET is_active = 0 WHERE uid = ?", (uid,))
                conn.commit()
                return c.rowcount > 0
        else:
            conn = self.pool.get_connection()
            c = conn.cursor()
            try:
                c.execute("UPDATE users SET is_active = FALSE WHERE uid = %s", (uid,))
                conn.commit()
                return c.rowcount > 0
            finally:
                c.close()
                conn.close()

    def get_auth_logs(self, limit=30):
        q = "SELECT id, user_uid, matched_name, timestamp, confidence, result, ip_address FROM auth_logs ORDER BY timestamp DESC LIMIT ?"
        if self.use_sqlite:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = [dict(r) for r in conn.execute(q, (limit,)).fetchall()]
                for r in rows:
                    r['timestamp'] = self._fmt_dt(r['timestamp'])
                return rows
        else:
            conn = self.pool.get_connection()
            c = conn.cursor(dictionary=True)
            try:
                c.execute(q.replace('?', '%s'), (limit,))
                rows = c.fetchall()
                for r in rows:
                    r['timestamp'] = self._fmt_dt(r.get('timestamp'))
                return rows
            finally:
                c.close()
                conn.close()
