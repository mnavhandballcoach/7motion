import os
import json
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from api.b2_client import upload_bytes

router = APIRouter()

CHUNK_DIR = "/tmp/chunks"
META_DIR = "/tmp/meta"

os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

@router.post("/upload_chunk")
async def upload_chunk(
    chunk: UploadFile,
    index: int = Form(...),
    filename: str = Form(...),
    email: str = Form(...),
    name: str = Form(...),
    start_time: int = Form(...),
    duration: int = Form(...)
):
    file_dir = os.path.join(CHUNK_DIR, filename)
    os.makedirs(file_dir, exist_ok=True)

    part_path = os.path.join(file_dir, f"{index}.part")
    with open(part_path, "wb") as f:
        f.write(await chunk.read())

    meta_path = os.path.join(META_DIR, f"{filename}.json")
    if not os.path.exists(meta_path):
        metadata = {
            "email": email,
            "name": name,
            "start_time": start_time,
            "duration": duration
        }
        with open(meta_path, "w") as m:
            json.dump(metadata, m)

    return {"status": "chunk_received", "index": index}


@router.get("/upload_chunk")
async def finish_upload(filename: str, finish: int = 0):
    if finish != 1:
        return JSONResponse({"error": "invalid_finish_flag"}, status_code=400)

    file_dir = os.path.join(CHUNK_DIR, filename)
    parts = sorted(os.listdir(file_dir), key=lambda x: int(x.replace(".part", "")))

    final_path = os.path.join(CHUNK_DIR, f"final_{filename}")
    with open(final_path, "wb") as final_file:
        for part in parts:
            with open(os.path.join(file_dir, part), "rb") as p:
                final_file.write(p.read())

    meta_path = os.path.join(META_DIR, f"{filename}.json")
    metadata = json.load(open(meta_path))

    upload_bytes(f"incoming/{filename}", open(final_path, "rb").read(), "video/mp4")

    json_name = filename.replace(".mp4", ".json")
    upload_bytes(f"incoming/{json_name}",
                 json.dumps(metadata).encode("utf-8"),
                 "application/json")

    return {"status": "upload_complete", "metadata": metadata}
