### routes/upload_routes.py

from ..functions import *
upload_bp, logger = get_bp("upload_bp", __name__)

@upload_bp.route("/uploads", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        abort(400, description="no file provided")
    os.makedirs(UPLOADS_HOME, exist_ok=True)
    name = f"{uuid.uuid4().hex[:8]}_{secure_filename(f.filename)}"
    dest = os.path.join(UPLOADS_HOME, name)
    f.save(dest)
    return jsonify({"path": dest, "name": f.filename, "size": os.path.getsize(dest)})
