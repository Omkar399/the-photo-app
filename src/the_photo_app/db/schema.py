"""SQLite schema definition and initialization."""

import sqlite3
import json
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: Path):
        """Initialize database connection."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init_db(self) -> None:
        """Initialize database schema if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Images table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    width INT,
                    height INT,
                    exif_json TEXT,
                    created_ts INT,
                    indexed_ts INT
                )
                """
            )

            # Image embeddings
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS image_vectors (
                    image_id TEXT PRIMARY KEY,
                    dim INT NOT NULL,
                    norm REAL,
                    FOREIGN KEY(image_id) REFERENCES images(image_id)
                )
                """
            )

            # Detected faces
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    face_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    person_label TEXT,
                    confidence REAL,
                    FOREIGN KEY(image_id) REFERENCES images(image_id)
                )
                """
            )

            # Person labels
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS people_labels (
                    person_label TEXT PRIMARY KEY,
                    note TEXT,
                    created_ts INT,
                    modified_ts INT
                )
                """
            )

            # Settings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

            # Optional: Captions & OCR with FTS
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS image_text USING fts5(
                    image_id UNINDEXED,
                    caption,
                    ocr
                )
                """
            )

            # Create indexes for faster queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_images_path ON images(path)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_faces_person_label ON faces(person_label)
                """
            )

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def upsert_image(
        self,
        image_id: str,
        path: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        exif_json: Optional[str] = None,
        created_ts: Optional[int] = None,
        indexed_ts: Optional[int] = None,
    ) -> None:
        """Insert or update image metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO images
                (image_id, path, width, height, exif_json, created_ts, indexed_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (image_id, path, width, height, exif_json, created_ts, indexed_ts),
            )
            conn.commit()

    def upsert_image_vector(
        self, image_id: str, dim: int, norm: Optional[float] = None
    ) -> None:
        """Insert or update image embedding metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO image_vectors (image_id, dim, norm)
                VALUES (?, ?, ?)
                """,
                (image_id, dim, norm),
            )
            conn.commit()

    def insert_face(
        self,
        face_id: str,
        image_id: str,
        bbox: str,  # JSON string [x1, y1, x2, y2]
        embedding: bytes,  # 512-D float32
        person_label: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Insert detected face."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO faces
                (face_id, image_id, bbox, embedding, person_label, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (face_id, image_id, bbox, embedding, person_label, confidence),
            )
            conn.commit()

    def get_faces_for_image(self, image_id: str):
        """Get all faces for an image."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM faces WHERE image_id = ?", (image_id,))
            return cursor.fetchall()

    def set_person_label(self, cluster_id: str, label: str, note: Optional[str] = None) -> None:
        """Set or update person label."""
        import time

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            ts = int(time.time())
            cursor.execute(
                """
                INSERT OR REPLACE INTO people_labels
                (person_label, note, created_ts, modified_ts)
                VALUES (?, ?, ?, ?)
                """,
                (label, note, ts, ts),
            )
            conn.commit()

    def update_face_label(self, face_id: str, person_label: str) -> None:
        """Update person label for a face."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE faces SET person_label = ? WHERE face_id = ?",
                (person_label, face_id),
            )
            conn.commit()

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a setting value."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def image_exists(self, image_id: str) -> bool:
        """Check if image exists in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM images WHERE image_id = ?", (image_id,))
            return cursor.fetchone() is not None

    def get_all_image_ids(self):
        """Get all image IDs (for indexing/validation)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT image_id FROM images")
            return [row[0] for row in cursor.fetchall()]

    def get_images_by_person_label(self, person_label: str):
        """Get all images containing a specific person."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT i.image_id, i.path
                FROM images i
                JOIN faces f ON i.image_id = f.image_id
                WHERE f.person_label = ?
                """,
                (person_label,),
            )
            return cursor.fetchall()
