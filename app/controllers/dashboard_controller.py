from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

router = APIRouter(tags=["Dashboard"])
_env   = Environment(loader=FileSystemLoader("templates"), cache_size=0)


@router.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_env.get_template("dashboard.html").render())
