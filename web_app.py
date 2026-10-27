"""Local HTTP API and static interface. Run via run_web.py (one worker only)."""

import secrets
from pathlib import Path

import torch
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.coding_problems import get_problems, new_problem
from src.model_loader import DEFAULT_MODEL_NAME
from src.web_service import ExperimentService, RunRequest, TextRunRequest
from src.walkthrough import WALKTHROUGH_VERSION, WALKTHROUGH_SENTENCE, WALKTHROUGH_SEED, WALKTHROUGH_PRESETS

ROOT = Path(__file__).resolve().parent


class NewProblemRequest(BaseModel):
    existing: list[str] = Field(default_factory=list, max_length=20)
    category: str = Field(default="all", max_length=40)


class WalkthroughRequest(BaseModel):
    # The client cannot submit a different sentence, model, seed or masks.
    model_config = {"extra": "forbid"}


def create_app(service=None):
    app = FastAPI(title="LLM visualization", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    service = service or ExperimentService()
    token = secrets.token_urlsafe(32)
    app.state.service = service

    @app.middleware("http")
    async def local_guard(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if not secrets.compare_digest(request.headers.get("X-Lab-Token", ""), token):
                return JSONResponse({"detail": "Refresh this local page and try again."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Upstream Svelte/D3 uses dynamic style attributes for layout/animation.
        # Permit those only in its isolated, same-origin visualization document;
        # the outer app retains its strict policy and cannot be framed.
        visualizer = request.url.path == "/static/transformer/index.html"
        styles = "'self' 'unsafe-inline'" if visualizer else "'self'"
        ancestors = "'self'" if visualizer else "'none'"
        fonts = "'self' data:" if visualizer else "'self'"
        response.headers["Content-Security-Policy"] = f"default-src 'self'; script-src 'self'; style-src {styles}; img-src 'self' data:; font-src {fonts}; connect-src 'self'; frame-ancestors {ancestors}; base-uri 'self'"
        return response

    @app.get("/")
    @app.get("/walkthrough")
    def index():
        return FileResponse(ROOT / "web" / "walkthrough.html")

    @app.get("/experiments")
    def experiments():
        return FileResponse(ROOT / "web" / "index.html")

    def active_for(kind):
        active = service.runs.get(service.active)
        if active and active["kind"] == kind:
            return service.active
        return None

    @app.get("/api/config")
    def config():
        return {
            "token": token, "model": DEFAULT_MODEL_NAME,
            "models": [{"id": DEFAULT_MODEL_NAME, "label": "Qwen2.5-Coder · 1.5B Instruct"}],
            "context_window": 32768, "generation_token_cap": None,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "problems": [p.model_dump() for p in get_problems()[:10]],
            "active_run": active_for("experiment"),
            "busy_elsewhere": service.active if service.active and not active_for("experiment") else None,
        }

    @app.get("/api/walkthrough/config")
    def walkthrough_config():
        return {"token": token, "version": WALKTHROUGH_VERSION, "sentence": WALKTHROUGH_SENTENCE,
                "model": "gpt2", "seed": WALKTHROUGH_SEED, "presets": WALKTHROUGH_PRESETS,
                "context_window": 1024, "generation_token_cap": None,
                "active_run": active_for("walkthrough"),
                "busy_elsewhere": service.active if service.active and not active_for("walkthrough") else None}

    @app.get("/api/problems")
    def problems(category: str = "all"):
        return {"problems": [p.model_dump() for p in get_problems(category)[:10]]}

    def start(request, kind):
        try:
            return {"id": service.start(request, kind)}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/runs", status_code=202)
    def create_run(request: RunRequest):
        if request.model_name != DEFAULT_MODEL_NAME:
            raise HTTPException(422, "The experiments page uses Qwen-Coder only. Use the GPT-2 walkthrough for the fixed sentence visualization.")
        return start(request, "experiment")

    @app.post("/api/walkthrough/ready", status_code=202)
    @app.post("/api/walkthrough/runs", status_code=202)
    def create_walkthrough(request: WalkthroughRequest):
        fixed = TextRunRequest(prompt=WALKTHROUGH_SENTENCE, seed=WALKTHROUGH_SEED,
                               variants=[{key: preset[key] for key in ("technique", "strength")} for preset in WALKTHROUGH_PRESETS])
        try:
            return {"id": service.ensure_walkthrough(fixed)}
        except RuntimeError as exc:
            raise HTTPException(409, "The shared model worker is busy. The walkthrough will prepare automatically when it becomes available.") from exc

    @app.post("/api/text-runs")
    def retired_text_run():
        raise HTTPException(410, "Editable text runs have been replaced by the fixed GPT-2 walkthrough at /api/walkthrough/runs.")

    @app.post("/api/problems/new")
    def generate_problem(request: NewProblemRequest):
        try:
            return new_problem(request.existing, request.category).model_dump()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/runs")
    def history():
        return service.history()

    @app.get("/api/runs/{run_id}")
    def run(run_id: str, include_pipelines: bool = False):
        try:
            return service.snapshot(run_id, include_pipelines=include_pipelines)
        except KeyError as exc:
            raise HTTPException(404, "Run not found.") from exc

    @app.get("/api/runs/{run_id}/pipeline")
    def pipeline(run_id: str, question: int = 0, variant: int = 0):
        try:
            return service.pipeline_pair(run_id, question, variant)
        except KeyError as exc:
            raise HTTPException(404, "Run not found.") from exc
        except IndexError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        run(run_id)
        service.cancel(run_id)
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app


app = create_app()
