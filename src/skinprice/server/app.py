"""Minimal web service: one HTML page and a small JSON API.

    GET  /                      the page
    GET  /api/health
    GET  /api/models            loaded models, versions and test metrics
    GET  /api/search?kind=&q=   autocomplete for items / stickers / charms
    POST /api/predict           {"listing": {...}} | {"url": "..."} | {"manual": {...}}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from skinprice.inference.predictor import Predictor
from skinprice.models.base import read_manifest
from skinprice.training import latest_dir

STATIC = Path(__file__).parent / "static"


def create_app(predictor: Predictor | None = None, models: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="CS2 skin price models", version="0.1.0")
    state: dict[str, Any] = {"predictor": predictor}

    def get_predictor() -> Predictor:
        if state["predictor"] is None:
            state["predictor"] = Predictor.from_workspace(models)
        return state["predictor"]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/models")
    def list_models() -> list[dict[str, Any]]:
        p = get_predictor()
        result = []
        for name, model in p.models.items():
            entry = {"name": name, "summary": model.summary, "version": p.versions.get(name)}
            try:
                manifest = read_manifest(latest_dir(name))
                entry["test_metrics"] = manifest["metrics"]["test"]
                entry["size_mb"] = manifest["size_mb"]
            except FileNotFoundError:
                pass
            result.append(entry)
        return result

    @app.get("/api/search")
    def search(kind: str = "items", q: str = "", limit: int = 15) -> list[str]:
        if kind not in {"items", "stickers", "keychains"}:
            raise HTTPException(400, "kind must be items, stickers or keychains")
        return get_predictor().catalog.search(kind, q, limit) if q else []

    @app.post("/api/predict")
    def predict(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return get_predictor().predict(payload)
        except (ValueError, KeyError) as error:
            raise HTTPException(422, str(error)) from error
        except OSError as error:  # network errors when fetching a listing
            raise HTTPException(502, f"Could not fetch the listing: {error}") from error

    return app
