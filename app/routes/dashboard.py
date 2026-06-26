from collections import Counter
from datetime import date

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

from model import UNKNOWN_DIR
from app.routes.attendance import read_records

router = APIRouter(tags=["Dashboard"])
_env   = Environment(loader=FileSystemLoader("templates"), cache_size=0)


def _render(name: str, **ctx) -> HTMLResponse:
    return HTMLResponse(_env.get_template(name).render(**ctx))


def _build_stats(records: list[dict]) -> dict:
    today_str  = date.today().isoformat()
    today      = [r for r in records if r["timestamp"].startswith(today_str)]
    today_cnt  = Counter(r["name"] for r in today)
    all_cnt    = Counter(r["name"] for r in records)
    most_seen  = today_cnt.most_common(1)[0][0] if today_cnt else None
    return {
        "today":         len(today),
        "total":         len(records),
        "unknown_count": len(list(UNKNOWN_DIR.glob("*.jpg"))),
        "most_seen":     most_seen,
        "chart": {
            "labels": list(all_cnt.keys()),
            "values": list(all_cnt.values()),
        },
    }


@router.get("/", response_class=HTMLResponse)
def index():
    records = read_records()
    stats   = _build_stats(records)
    return _render("dashboard.html",
        stats          = stats,
        chart_data     = stats["chart"],
        recent         = list(reversed(records))[:10],
        records        = list(reversed(records)),
        unknown_images = sorted([f.name for f in UNKNOWN_DIR.glob("*.jpg")], reverse=True),
    )
