import json
import sqlite3
from pathlib import Path

from .models import Fingerprint

DEFAULT_DB_PATH = Path.home() / ".idiolect" / "fingerprints.db"


class FingerprintStore:
    """SQLite-backed storage for enrolled author fingerprints."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the fingerprints table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fingerprints (
                    label TEXT PRIMARY KEY,
                    fingerprint_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    word_count INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def enroll(self, fingerprint: Fingerprint) -> None:
        """Store a fingerprint under its label. Overwrites if label exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO fingerprints (label, fingerprint_json, created_at, word_count)
                VALUES (?, ?, ?, ?)
                """,
                (
                    fingerprint.label,
                    fingerprint.to_json(),
                    fingerprint.created_at,
                    fingerprint.word_count,
                ),
            )
            conn.commit()

    def get(self, label: str) -> Fingerprint | None:
        """Retrieve a fingerprint by label."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fingerprint_json FROM fingerprints WHERE label = ?", (label,))
            row = cursor.fetchone()
            if row:
                return Fingerprint.from_json(row[0])
            return None

    def list_all(self) -> list[str]:
        """List all enrolled fingerprint labels."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT label FROM fingerprints ORDER BY label")
            return [row[0] for row in cursor.fetchall()]

    def delete(self, label: str) -> bool:
        """Delete a fingerprint by label. Returns True if found and deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM fingerprints WHERE label = ?", (label,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def get_all(self) -> list[Fingerprint]:
        """Retrieve all enrolled fingerprints."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fingerprint_json FROM fingerprints ORDER BY label")
            return [Fingerprint.from_json(row[0]) for row in cursor.fetchall()]
