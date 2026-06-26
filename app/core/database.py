from pathlib import Path
from tinydb import TinyDB, Query

_db = TinyDB(Path("db.json"), indent=2)

orgs_table       = _db.table("organizations")
members_table    = _db.table("members")
attendance_table = _db.table("attendance")

OrgQuery    = Query()
MemberQuery = Query()
AttQuery    = Query()
