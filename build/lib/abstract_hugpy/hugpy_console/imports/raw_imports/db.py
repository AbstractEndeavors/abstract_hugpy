# src/db.py
from .imports import *

# Build DATABASE_URL from env vars
env_vars = [
    "ABSTRACT_DATABASE_USER",
    "ABSTRACT_DATABASE_PORT",
    "ABSTRACT_DATABASE_DBNAME",
    "ABSTRACT_DATABASE_HOST",
    "ABSTRACT_DATABASE_PASSWORD",
]

def safe_get_env(key):
    val = get_env_value(key)
    return val if val is not None else ""

env_js = {v.split("_")[-1]: safe_get_env(v) for v in env_vars}
env_js["HOST"] = env_js.get("HOST") or "23.126.105.154"

DATABASE_URL = (
    f"postgresql://{env_js['USER']}:{env_js['PASSWORD']}@"
    f"{env_js['HOST']}:{env_js['PORT']}/{env_js['DBNAME']}"
)

print(f"[DB] Using {DATABASE_URL}")

engine = create_engine(DATABASE_URL, future=True, pool_size=10, max_overflow=20)
metadata = MetaData()

VIDEOSTABLE = Table(
    "videos", metadata,
    Column("id", Integer, primary_key=True),
    Column("video_id", String, unique=True, nullable=False),
    Column("info", JSONB),
    Column("metadata", JSONB),
    Column("whisper", JSONB),
    Column("captions", JSONB),
    Column("thumbnails", JSONB),
    Column("total_info", JSONB),
    Column("aggregated", JSONB),
    Column("seodata", JSONB),
    Column("audio_path", String),
    Column("audio_format", String),
    Column("created_at", TIMESTAMP, server_default=text("NOW()")),
    Column("updated_at", TIMESTAMP, server_default=text("NOW()")),
)

def init_db():
    metadata.create_all(engine)

def sanitize_output(record: dict) -> dict:
    if "audio" in record:
        record["audio"] = f"<{len(record['audio'])} bytes>" if record["audio"] else None
    return record

def upsert_video(video_id: str, **fields):
    stmt = insert(VIDEOSTABLE).values(video_id=video_id, **fields)
    stmt = stmt.on_conflict_do_update(
        index_elements=["video_id"],
        set_={**fields, "updated_at": text("NOW()")}
    )
    with engine.begin() as conn:
        conn.execute(stmt)

def get_video_record(video_id: str, hide_audio: bool = True):
    with engine.begin() as conn:
        row = conn.execute(select(VIDEOSTABLE).where(VIDEOSTABLE.c.video_id == video_id)).first()
        if not row:
            return None
        record = dict(row._mapping)
        if hide_audio and "audio" in record:
            record["audio"] = f"<{len(record['audio'])} bytes>" if record["audio"] else None
        return record

# Initialize at import
init_db()

def sanitize_output(record: dict) -> dict:
    if "audio" in record:
        record["audio"] = f"<{len(record['audio'])} bytes>" if record["audio"] else None
    return record

def upsert_video(video_id: str, **fields):
    """Insert or update a video record."""
    stmt = insert(VIDEOSTABLE).values(video_id=video_id, **fields)
    stmt = stmt.on_conflict_do_update(
        index_elements=["video_id"],
        set_={**fields, "updated_at": text("NOW()")}
    )
    with engine.begin() as conn:
        conn.execute(stmt)

def get_video_record(video_id: str, hide_audio: bool = True):
    with engine.begin() as conn:
        row = conn.execute(select(VIDEOSTABLE).where(VIDEOSTABLE.c.video_id == video_id)).first()
        if not row:
            return None
        record = dict(row._mapping)
        if hide_audio and "audio" in record:
            # Replace huge binary blob with a short placeholder
            record["audio"] = f"<{len(record['audio'])} bytes>" if record["audio"] else None
        return record
