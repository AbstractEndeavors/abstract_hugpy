from .imports import *
def model_status(model: dict) -> dict:
    destination = route_destination(model)              # was model_destination(...)
    marker = os.path.join(destination, HUGPY_MARKER)    # was install_marker(...)
    if os.path.exists(marker):
        status = "installed"
    elif os.path.exists(destination) and os.listdir(destination):
        status = "partial"
    else:
        status = "not_installed"
    return {"status": status, "destination": destination, "installed_marker": marker}

def write_install_marker(destination: str, model_key: str, model: dict[str, Any]) -> None:
    marker = install_marker(destination)
    payload = {
        "model_key": model_key,
        "hub_id": model.get("hub_id"),
        "framework": model.get("framework"),
        "task": model.get("task"),
        "filename": model.get("filename"),
        "include": model.get("include"),
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(marker, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))



