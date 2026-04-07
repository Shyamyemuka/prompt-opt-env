"""FastAPI app for PromptOptEnv."""

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
