
import os

import pytest
from sqlalchemy import delete

os.environ["DATABASE_URL"] = (
    "postgresql+psycopg://eki_user:eki_password"
    "@localhost:5432/eki_db_test"
)


@pytest.fixture(autouse=True)
def clean_test_database():
    yield

    from app.db.session import SessionLocal
    from app.models.document import Document, DocumentChunk

    db = SessionLocal()

    try:
        db.execute(delete(DocumentChunk))
        db.execute(delete(Document))
        db.commit()
    finally:
        db.close()

