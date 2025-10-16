# src/db.py
from .imports import *


def return_num_str(obj):
    return int(obj) if is_number(obj) else obj


# Build DATABASE_URL from env vars
emv_vars = """ABSTRACT_DATABASE_USER
ABSTRACT_DATABASE_PORT
ABSTRACT_DATABASE_DBNAME
ABSTRACT_DATABASE_HOST
ABSTRACT_DATABASE_PASSWORD""".split("\n")

emv_js = {ev.split("_")[-1]: return_num_str(get_env_value(ev)) for ev in emv_vars}

DATABASE_URL = (
    f"postgresql://{emv_js.get('USER')}:{emv_js.get('PASSWORD')}@"
    f"{emv_js.get('HOST')}:{emv_js.get('PORT')}/{emv_js.get('DBNAME')}"
)

engine = create_engine(DATABASE_URL, future=True, pool_size=10, max_overflow=20)
metadata = MetaData()

videos = Table(
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
    Column("audio", LargeBinary),
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
    """Insert or update a video record."""
    stmt = insert(videos).values(video_id=video_id, **fields)
    stmt = stmt.on_conflict_do_update(
        index_elements=["video_id"],
        set_={**fields, "updated_at": text("NOW()")}
    )
    with engine.begin() as conn:
        conn.execute(stmt)

def get_video_record(video_id: str, hide_audio: bool = True):
    with engine.begin() as conn:
        row = conn.execute(select(videos).where(videos.c.video_id == video_id)).first()
        if not row:
            return None
        record = dict(row._mapping)
        if hide_audio and "audio" in record:
            # Replace huge binary blob with a short placeholder
            record["audio"] = f"<{len(record['audio'])} bytes>" if record["audio"] else None
        return record
