import pytest

from app import server


@pytest.mark.parametrize("path", ["/", "/_dash-layout"])
def test_dashboard_startup(path):
    with server.test_client() as client:
        response = client.get(path)

    assert response.status_code == 200
