"""
Database Module for DemandWise
==============================
SQLite database operations for user authentication, business profiles,
and multi-store management using parameterized queries.
"""

import os
import sqlite3
from typing import Dict, Any, List, Optional

def get_default_db_path() -> str:
    candidates = [
        os.path.abspath(os.path.join('database', 'demandwise.db')),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'demandwise.db'),
        os.path.abspath(os.path.join('..', 'database', 'demandwise.db'))
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

DEFAULT_DB_PATH = get_default_db_path()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Returns a SQLite connection with row_factory set to sqlite3.Row."""
    target_path = db_path or get_default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes tables for users, business profiles, and stores.
    Also applies safe migrations for new columns on existing databases.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS business_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                business_name TEXT NOT NULL,
                owner_name TEXT NOT NULL,
                currency TEXT DEFAULT 'USD ($)',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                store_name TEXT NOT NULL,
                store_code TEXT,
                store_type TEXT,
                location TEXT,
                dataset_path TEXT,
                is_demo INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # Safe migrations: add new dataset metadata columns if missing
        existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(stores)").fetchall()}
        new_columns = {
            'dataset_filename': 'TEXT',
            'dataset_upload_date': 'TEXT',
            'dataset_row_count': 'INTEGER',
            'dataset_date_min': 'TEXT',
            'dataset_date_max': 'TEXT',
            'dataset_revenue_calculated': 'INTEGER DEFAULT 0',
            'dataset_column_warnings': 'TEXT',
        }
        for col_name, col_type in new_columns.items():
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE stores ADD COLUMN {col_name} {col_type};")

        conn.commit()


# ---------------------------------------------------------------------------
# User Operations
# ---------------------------------------------------------------------------

def create_user(username: str, email: str, password_hash: str, salt: str, db_path: Optional[str] = None) -> int:
    """Creates a new user record. Returns user_id."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)",
            (username.strip(), email.strip().lower(), password_hash, salt)
        )
        conn.commit()
        return cursor.lastrowid


def get_user_by_username(username: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves user by username."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves user by id."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Business Profile Operations
# ---------------------------------------------------------------------------

def get_business_profile(user_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves business profile for a given user."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM business_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def upsert_business_profile(
    user_id: int,
    business_name: str,
    owner_name: str,
    currency: str = "USD ($)",
    db_path: Optional[str] = None
) -> None:
    """Inserts or updates business profile for a user."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO business_profiles (user_id, business_name, owner_name, currency)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                business_name = excluded.business_name,
                owner_name = excluded.owner_name,
                currency = excluded.currency;
        """, (user_id, business_name.strip(), owner_name.strip(), currency.strip()))
        conn.commit()


# ---------------------------------------------------------------------------
# Store Operations
# ---------------------------------------------------------------------------

def get_stores_for_user(user_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns all stores owned by user."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stores WHERE user_id = ? ORDER BY id ASC", (user_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_store_by_id(store_id: int, user_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a store by ID, ensuring ownership by user_id."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stores WHERE id = ? AND user_id = ?", (store_id, user_id))
        row = cursor.fetchone()
        return dict(row) if row else None


def add_store(
    user_id: int,
    store_name: str,
    store_code: Optional[str] = None,
    store_type: Optional[str] = None,
    location: Optional[str] = None,
    dataset_path: Optional[str] = None,
    is_demo: int = 0,
    db_path: Optional[str] = None
) -> int:
    """Adds a new store for user. Returns store_id."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO stores (user_id, store_name, store_code, store_type, location, dataset_path, is_demo)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, store_name.strip(), store_code, store_type, location, dataset_path, is_demo)
        )
        conn.commit()
        return cursor.lastrowid


def update_store(
    store_id: int,
    user_id: int,
    store_name: str,
    store_code: Optional[str] = None,
    store_type: Optional[str] = None,
    location: Optional[str] = None,
    dataset_path: Optional[str] = None,
    db_path: Optional[str] = None
) -> bool:
    """Updates a store's profile fields owned by user.
    Does NOT clear dataset metadata columns; use update_store_dataset_meta() for those.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE stores
            SET store_name = ?, store_code = ?, store_type = ?, location = ?, dataset_path = ?
            WHERE id = ? AND user_id = ?
            """,
            (store_name.strip(), store_code, store_type, location, dataset_path, store_id, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0


def update_store_dataset_meta(
    store_id: int,
    user_id: int,
    dataset_path: str,
    dataset_filename: str,
    dataset_upload_date: str,
    dataset_row_count: int,
    dataset_date_min: Optional[str],
    dataset_date_max: Optional[str],
    dataset_revenue_calculated: int = 0,
    dataset_column_warnings: Optional[str] = None,
    db_path: Optional[str] = None
) -> bool:
    """
    Atomically updates dataset path and all dataset metadata columns for a store.
    This is used after a successful validated upload so that all fields update together.
    Ownership by user_id is enforced.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE stores
            SET dataset_path = ?,
                dataset_filename = ?,
                dataset_upload_date = ?,
                dataset_row_count = ?,
                dataset_date_min = ?,
                dataset_date_max = ?,
                dataset_revenue_calculated = ?,
                dataset_column_warnings = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                dataset_path,
                dataset_filename,
                dataset_upload_date,
                dataset_row_count,
                dataset_date_min,
                dataset_date_max,
                dataset_revenue_calculated,
                dataset_column_warnings,
                store_id,
                user_id
            )
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_store(store_id: int, user_id: int, db_path: Optional[str] = None) -> bool:
    """Deletes a store owned by user."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM stores WHERE id = ? AND user_id = ?", (store_id, user_id))
        conn.commit()
        return cursor.rowcount > 0


def seed_demo_stores_for_user(user_id: int, data_dir: Optional[str] = None, db_path: Optional[str] = None) -> List[int]:
    """
    Seeds the 5 default demo stores for a user if they don't have stores yet.
    Returns list of newly created store IDs.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_data_dir = data_dir or os.path.join(base_dir, 'data')

    demo_definitions = [
        {
            "name": "Store_01_Central",
            "code": "STR-01",
            "type": "Flagship Downtown",
            "location": "Central Business District",
            "file": "DemandWise_Store_01_Central.xlsx"
        },
        {
            "name": "Store_02_Market",
            "code": "STR-02",
            "type": "Urban Supermarket",
            "location": "City Market Plaza",
            "file": "DemandWise_Store_02_Market.xlsx"
        },
        {
            "name": "Store_03_Residential",
            "code": "STR-03",
            "type": "Neighborhood Store",
            "location": "Green Valley Residential",
            "file": "DemandWise_Store_03_Residential.xlsx"
        },
        {
            "name": "Store_04_Suburban",
            "code": "STR-04",
            "type": "Suburban Retail",
            "location": "West Pines Commercial Zone",
            "file": "DemandWise_Store_04_Suburban.xlsx"
        },
        {
            "name": "Store_05_HighStreet",
            "code": "STR-05",
            "type": "High Street Mall",
            "location": "High Street Shopping District",
            "file": "DemandWise_Store_05_HighStreet.xlsx"
        }
    ]

    created_ids = []
    existing = get_stores_for_user(user_id, db_path=db_path)
    existing_names = {s['store_name'] for s in existing}

    for demo in demo_definitions:
        if demo["name"] not in existing_names:
            full_path = os.path.join(target_data_dir, demo["file"])
            rel_path = os.path.join('data', demo["file"]) if os.path.exists(full_path) else demo["file"]
            sid = add_store(
                user_id=user_id,
                store_name=demo["name"],
                store_code=demo["code"],
                store_type=demo["type"],
                location=demo["location"],
                dataset_path=rel_path,
                is_demo=1,
                db_path=db_path
            )
            created_ids.append(sid)

    return created_ids
