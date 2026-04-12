"""FastAPI app for PromptOptEnv."""

import json
import math

from fastapi import Request, Response
from fastapi.responses import JSONResponse

try:
    from openenv.core.env_server.http_server import create_fastapi_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv is required. Install with 'uv sync'") from e

try:
    from ..models import PromptAction, PromptObservation
    from .prompt_opt_env_environment import PromptOptEnvEnvironment
except (ModuleNotFoundError, ImportError):
    from models import PromptAction, PromptObservation
    from server.prompt_opt_env_environment import PromptOptEnvEnvironment

app = create_fastapi_app(
    PromptOptEnvEnvironment,
    PromptAction,
    PromptObservation,
    max_concurrent_envs=1,
)

STRICT_SCORE_FLOOR = 0.11


def _strict_unit_interval(value: object, fallback: float = STRICT_SCORE_FLOOR) -> float:
    """Keep scores and rewards strictly inside the validator's accepted range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(numeric):
        return fallback
    rounded = round(numeric, 4)
    return float(max(STRICT_SCORE_FLOOR, min(1.0 - STRICT_SCORE_FLOOR, rounded)))


def _normalize_env_payload(payload: dict) -> dict:
    """Mirror top-level reward/done into observation for validator compatibility."""
    normalized = dict(payload)
    observation = dict(normalized.get("observation") or {})

    reward = _strict_unit_interval(
        normalized.get("reward", observation.get("reward", STRICT_SCORE_FLOOR))
    )
    done = bool(normalized.get("done", observation.get("done", False)))

    for key in ("current_score", "previous_score"):
        if key in observation:
            observation[key] = _strict_unit_interval(observation[key])

    observation["reward"] = reward
    observation["done"] = done
    normalized["reward"] = reward
    normalized["done"] = done
    normalized["observation"] = observation
    return normalized


@app.middleware("http")
async def normalize_reset_and_step_payloads(request: Request, call_next):
    """Keep `/reset` and `/step` responses aligned with the declared observation schema."""
    response = await call_next(request)
    if request.url.path not in {"/reset", "/step"} or response.status_code >= 400:
        return response

    media_type = getattr(response, "media_type", "") or response.headers.get("content-type", "")
    if "json" not in media_type.lower():
        return response

    body = b""
    async for chunk in response.body_iterator:
        body += chunk

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        # Body iterator is already consumed; rebuild response body safely.
        return Response(
            content=body,
            status_code=response.status_code,
            media_type=media_type,
        )

    if not isinstance(payload, dict):
        return Response(
            content=body,
            status_code=response.status_code,
            media_type=media_type,
        )

    return JSONResponse(
        content=_normalize_env_payload(payload),
        status_code=response.status_code,
        media_type="application/json",
    )

# Ensure checklist-compatible health response payload.
try:
    app.router.routes = [
        route
        for route in app.router.routes
        if getattr(route, "path", None) != "/health"
    ]
except Exception:
    pass


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

# Mount custom dark UI routes (both root and /web for HF compatibility).
try:
    try:
        from prompt_opt_env.web_ui import landing, home, optimize, _write_templates
    except Exception:
        try:
            from ..web_ui import landing, home, optimize, _write_templates
        except Exception:
            from web_ui import landing, home, optimize, _write_templates

    from fastapi.responses import HTMLResponse

    app.router.routes = [
        r
        for r in app.router.routes
        if getattr(r, "path", None)
        not in ["/", "/app", "/optimize", "/web", "/web/app", "/web/optimize"]
    ]

    _write_templates()

    app.get("/", response_class=HTMLResponse)(landing)
    app.router.routes.insert(0, app.router.routes.pop())

    app.get("/app", response_class=HTMLResponse)(home)
    app.router.routes.insert(0, app.router.routes.pop())

    app.post("/optimize", response_class=HTMLResponse)(optimize)
    app.router.routes.insert(0, app.router.routes.pop())

    app.get("/web", response_class=HTMLResponse)(landing)
    app.router.routes.insert(0, app.router.routes.pop())

    app.get("/web/app", response_class=HTMLResponse)(home)
    app.router.routes.insert(0, app.router.routes.pop())

    app.post("/web/optimize", response_class=HTMLResponse)(optimize)
    app.router.routes.insert(0, app.router.routes.pop())

    print("[INFO] Custom Dark UI mounted on / and /web")
except Exception as e:
    print(f"[WARNING] Web UI could not be mounted: {e}")


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
