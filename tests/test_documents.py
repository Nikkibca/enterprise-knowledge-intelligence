from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_list_documents():
    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_delete_document():
    response = client.delete("/documents/999999999")

    assert response.status_code == 404

    assert response.json() == {
        "detail": (
            "Document with id 999999999 "
            "was not found."
        )
    }


def test_delete_existing_document():
    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert "documents" in data
    assert isinstance(data["documents"], list)

    documents = data["documents"]

    if not documents:
        return

    document_id = documents[0]["id"]

    delete_response = client.delete(
        f"/documents/{document_id}"
    )

    assert delete_response.status_code == 200

    delete_data = delete_response.json()

    assert delete_data["id"] == document_id
    assert delete_data["message"].endswith(
        "was deleted successfully."
    )