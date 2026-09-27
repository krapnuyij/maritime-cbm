from fastapi.testclient import TestClient

from maritime_cbm.api.app import create_app


def test_demo_static_page_redirect_assets_and_openapi_visibility() -> None:
    client = TestClient(create_app())

    redirect = client.get("/demo", follow_redirects=False)
    page = client.get("/demo/")
    stylesheet = client.get("/demo/demo.css")
    script = client.get("/demo/demo.js")
    openapi = client.get("/openapi.json")

    assert redirect.status_code == 307
    assert redirect.headers["location"].endswith("/demo/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert 'id="sensor-form"' in page.text
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert script.status_code == 200
    script_media_type = script.headers["content-type"].split(";", maxsplit=1)[0]
    assert script_media_type in {"application/javascript", "text/javascript"}
    assert "/v1/condition/predict" in script.text
    assert "/v1/alert/evaluate" in script.text
    assert openapi.status_code == 200
    assert all(not path.startswith("/demo") for path in openapi.json()["paths"])
