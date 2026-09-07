import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import AuthorProfile, AuthorSample, Fingerprint
from .profiling import aggregate_fingerprints

DEFAULT_DB_PATH = Path.home() / ".idiolect" / "fingerprints.db"


class FingerprintStore:
    """SQLite-backed storage for enrolled author profiles and multi-sample fingerprints."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a resilient SQLite connection with timeout and busy_timeout configured."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def _init_db(self) -> None:
        """Create tables for fingerprints and author samples, migrating schema if needed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fingerprints (
                    label TEXT PRIMARY KEY,
                    fingerprint_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS author_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author_label TEXT NOT NULL,
                    sample_label TEXT NOT NULL,
                    fingerprint_json TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    enrolled_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_author_samples_author "
                "ON author_samples(author_label)"
            )

            # Migration: Ensure sample_count exists in fingerprints
            cursor.execute("PRAGMA table_info(fingerprints)")
            cols = [info[1] for info in cursor.fetchall()]
            if "sample_count" not in cols:
                cursor.execute(
                    "ALTER TABLE fingerprints ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 1"
                )

            # Migration: Backfill author_samples for any legacy profiles missing sample rows
            cursor.execute(
                """
                SELECT f.label, f.fingerprint_json, f.created_at, f.word_count
                FROM fingerprints f
                LEFT JOIN author_samples s ON f.label = s.author_label
                WHERE s.id IS NULL
                """
            )
            for row in cursor.fetchall():
                lbl, fp_json, cr_at, wc = row
                cursor.execute(
                    """
                    INSERT INTO author_samples
                    (author_label, sample_label, fingerprint_json, word_count, enrolled_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (lbl, "Initial Sample", fp_json, wc, cr_at),
                )
            conn.commit()

    def enroll_sample(
        self,
        author_label: str,
        fingerprint: Fingerprint,
        sample_label: str | None = None,
        replace: bool = False,
        recency_decay: float = 0.90,
        max_samples: int = 20,
    ) -> Fingerprint:
        """Enroll a sample for an author and recalculate rolling composite baseline.

        Args:
            author_label: The author's unique name/label.
            fingerprint: The sample's Fingerprint.
            sample_label: Optional label for this specific sample (e.g. filename).
            replace: If True, clear all prior samples and start a fresh profile.
            recency_decay: Exponential decay parameter for rolling average (0.1 to 1.0).
            max_samples: Maximum number of recent samples to keep in the rolling window.

        Returns:
            The newly synthesized composite Fingerprint.
        """
        sample_name = sample_label or fingerprint.source_path or fingerprint.label or "Sample"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if replace:
                cursor.execute("DELETE FROM author_samples WHERE author_label = ?", (author_label,))

            cursor.execute(
                """
                INSERT INTO author_samples
                (author_label, sample_label, fingerprint_json, word_count, enrolled_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    author_label,
                    Path(sample_name).name,
                    fingerprint.to_json(),
                    fingerprint.word_count,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            cursor.execute(
                "SELECT fingerprint_json FROM author_samples "
                "WHERE author_label = ? ORDER BY id ASC",
                (author_label,),
            )
            all_fps = [Fingerprint.from_json(r[0]) for r in cursor.fetchall()]
            composite_fp = aggregate_fingerprints(
                all_fps,
                label=author_label,
                recency_decay=recency_decay,
                max_samples=max_samples,
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO fingerprints
                (label, fingerprint_json, created_at, word_count, sample_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    author_label,
                    composite_fp.to_json(),
                    composite_fp.created_at,
                    composite_fp.word_count,
                    composite_fp.sample_count,
                ),
            )
            conn.commit()
            return composite_fp

    def enroll_samples(
        self,
        author_label: str,
        samples: list[tuple[str, Fingerprint]],
        replace: bool = False,
        recency_decay: float = 0.90,
        max_samples: int = 20,
    ) -> Fingerprint:
        """Enroll multiple samples for an author and recalculate rolling composite baseline.

        Args:
            author_label: The author's unique name/label.
            samples: List of (sample_label, fingerprint) tuples.
            replace: If True, clear all prior samples and start fresh.
            recency_decay: Exponential decay parameter for rolling average.
            max_samples: Maximum number of recent samples in rolling window.

        Returns:
            The newly synthesized composite Fingerprint.
        """
        if not samples:
            raise ValueError("No samples provided for enrollment.")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if replace:
                cursor.execute("DELETE FROM author_samples WHERE author_label = ?", (author_label,))

            now_str = datetime.now(timezone.utc).isoformat()
            for sample_name, fp in samples:
                cursor.execute(
                    """
                    INSERT INTO author_samples
                    (author_label, sample_label, fingerprint_json, word_count, enrolled_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (author_label, Path(sample_name).name, fp.to_json(), fp.word_count, now_str),
                )

            cursor.execute(
                "SELECT fingerprint_json FROM author_samples "
                "WHERE author_label = ? ORDER BY id ASC",
                (author_label,),
            )
            all_fps = [Fingerprint.from_json(r[0]) for r in cursor.fetchall()]
            composite_fp = aggregate_fingerprints(
                all_fps,
                label=author_label,
                recency_decay=recency_decay,
                max_samples=max_samples,
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO fingerprints
                (label, fingerprint_json, created_at, word_count, sample_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    author_label,
                    composite_fp.to_json(),
                    composite_fp.created_at,
                    composite_fp.word_count,
                    composite_fp.sample_count,
                ),
            )
            conn.commit()
            return composite_fp

    def enroll(
        self,
        fingerprint: Fingerprint,
        sample_label: str | None = None,
        replace: bool = False,
    ) -> Fingerprint:
        """Store or update a fingerprint. Backward-compatible with single-fingerprint calls."""
        return self.enroll_sample(
            author_label=fingerprint.label,
            fingerprint=fingerprint,
            sample_label=sample_label,
            replace=replace,
        )

    def get(self, label: str) -> Fingerprint | None:
        """Retrieve the composite fingerprint by author label."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fingerprint_json FROM fingerprints WHERE label = ?", (label,))
            row = cursor.fetchone()
            if row:
                return Fingerprint.from_json(row[0])
            return None

    def get_samples(self, author_label: str) -> list[AuthorSample]:
        """Retrieve all individual enrolled samples for an author."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, author_label, sample_label, fingerprint_json, word_count, enrolled_at
                FROM author_samples
                WHERE author_label = ?
                ORDER BY id ASC
                """,
                (author_label,),
            )
            samples = []
            for row in cursor.fetchall():
                sid, albl, slbl, fp_json, wc, eat = row
                fp = Fingerprint.from_json(fp_json)
                samples.append(
                    AuthorSample(
                        id=sid,
                        author_label=albl,
                        sample_label=slbl,
                        word_count=wc,
                        enrolled_at=eat,
                        fingerprint=fp,
                    )
                )
            return samples

    def get_profile(self, author_label: str) -> AuthorProfile | None:
        """Retrieve complete AuthorProfile using a single database connection."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT fingerprint_json FROM fingerprints WHERE label = ?", (author_label,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            comp_fp = Fingerprint.from_json(row[0])

            cursor.execute(
                """
                SELECT id, author_label, sample_label, fingerprint_json, word_count, enrolled_at
                FROM author_samples
                WHERE author_label = ?
                ORDER BY id ASC
                """,
                (author_label,),
            )
            samples = []
            for s_row in cursor.fetchall():
                sid, albl, slbl, fp_json, wc, eat = s_row
                samples.append(
                    AuthorSample(
                        id=sid,
                        author_label=albl,
                        sample_label=slbl,
                        word_count=wc,
                        enrolled_at=eat,
                        fingerprint=Fingerprint.from_json(fp_json),
                    )
                )
            return AuthorProfile(
                label=author_label,
                composite_fingerprint=comp_fp,
                samples=samples,
            )

    def list_all(self) -> list[str]:
        """List all enrolled fingerprint labels."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT label FROM fingerprints ORDER BY label")
            return [row[0] for row in cursor.fetchall()]

    def list_profiles(self) -> list[AuthorProfile]:
        """Retrieve all enrolled author profiles in a single batched query."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT label, fingerprint_json FROM fingerprints ORDER BY label")
            fp_rows = cursor.fetchall()
            if not fp_rows:
                return []

            cursor.execute(
                """
                SELECT id, author_label, sample_label, fingerprint_json, word_count, enrolled_at
                FROM author_samples
                ORDER BY id ASC
                """
            )
            sample_rows = cursor.fetchall()

        samples_by_author: dict[str, list[AuthorSample]] = {}
        for row in sample_rows:
            sid, albl, slbl, fp_json, wc, eat = row
            s = AuthorSample(
                id=sid,
                author_label=albl,
                sample_label=slbl,
                word_count=wc,
                enrolled_at=eat,
                fingerprint=Fingerprint.from_json(fp_json),
            )
            samples_by_author.setdefault(albl, []).append(s)

        profiles = []
        for lbl, fp_json in fp_rows:
            comp_fp = Fingerprint.from_json(fp_json)
            profiles.append(
                AuthorProfile(
                    label=lbl,
                    composite_fingerprint=comp_fp,
                    samples=samples_by_author.get(lbl, []),
                )
            )
        return profiles

    def delete(self, label: str) -> bool:
        """Delete an author profile and all associated samples.

        Returns True if found and deleted.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM fingerprints WHERE label = ?", (label,))
            deleted = cursor.rowcount > 0
            cursor.execute("DELETE FROM author_samples WHERE author_label = ?", (label,))
            conn.commit()
            return deleted

    def delete_sample(
        self,
        author_label: str,
        sample_id: int,
        recency_decay: float = 0.90,
        max_samples: int = 20,
    ) -> Fingerprint | None:
        """Delete a specific sample by ID and re-aggregate remaining samples.

        Returns updated composite Fingerprint, or None if author was deleted (0 samples left)
        or if sample ID was not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM author_samples WHERE id = ? AND author_label = ?",
                (sample_id, author_label),
            )
            if cursor.rowcount == 0:
                return None

            cursor.execute(
                "SELECT fingerprint_json FROM author_samples "
                "WHERE author_label = ? ORDER BY id ASC",
                (author_label,),
            )
            rows = cursor.fetchall()
            if not rows:
                cursor.execute("DELETE FROM fingerprints WHERE label = ?", (author_label,))
                conn.commit()
                return None

            all_fps = [Fingerprint.from_json(r[0]) for r in rows]
            composite_fp = aggregate_fingerprints(
                all_fps,
                label=author_label,
                recency_decay=recency_decay,
                max_samples=max_samples,
            )
            cursor.execute(
                """
                INSERT OR REPLACE INTO fingerprints
                (label, fingerprint_json, created_at, word_count, sample_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    author_label,
                    composite_fp.to_json(),
                    composite_fp.created_at,
                    composite_fp.word_count,
                    composite_fp.sample_count,
                ),
            )
            conn.commit()
            return composite_fp

    def get_all(self) -> list[Fingerprint]:
        """Retrieve all enrolled fingerprints."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fingerprint_json FROM fingerprints ORDER BY label")
            return [Fingerprint.from_json(row[0]) for row in cursor.fetchall()]
