# routes.py
@router.post("/run")
async def run(req_body: dict):
    """Generic single-shot endpoint. Validates against the runner's request type."""
    model_key = req_body.get("model_key")
    if not model_key:
        return JSONResponse({"ok": False, "error": "missing model_key"}, 400)

    try:
        runner = runner_for(model_key)
        req = runner.request_type(**req_body)
        result = await runner.run(req)
        return result.model_dump()
    except KeyError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, 404)
    except ValidationError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, 422)


@router.post("/stream")
async def stream(req_body: dict, http: Request):
    """Streaming endpoint — only works for runners that implement .stream()."""
    runner = runner_for(req_body["model_key"])
    req = runner.request_type(**req_body)

    cancel = asyncio.Event()
    async with _CANCELS_LOCK:
        _CANCELS[req.request_id] = cancel

    async def event_stream():
        watcher = asyncio.create_task(_watch_disconnect(http, cancel))
        try:
            async for event in runner.stream(req, cancel_event=cancel):
                yield f"data: {event.model_dump_json()}\n\n"
        except NotImplementedError:
            yield f"data: {ErrorEvent(request_id=req.request_id, message='streaming not supported').model_dump_json()}\n\n"
        finally:
            watcher.cancel()
            async with _CANCELS_LOCK:
                _CANCELS.pop(req.request_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
