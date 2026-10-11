import os
import json
import shutil
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
    # ID fixo baseado no nome do ficheiro
    video_id = filename.replace(".mp4", "").replace(".mov", "")

    # Pasta dos chunks
    file_dir = os.path.join(CHUNK_DIR, video_id)
    os.makedirs(file_dir, exist_ok=True)

    # Guardar chunk
    part_path = os.path.join(file_dir, f"{index}.part")
    with open(part_path, "wb") as f:
        f.write(await chunk.read())

    # Guardar metadata (apenas 1 vez)
    meta_path = os.path.join(META_DIR, f"{video_id}.json")
    if not os.path.exists(meta_path):
        metadata = {
            "email": email,
            "name": name,
            "start": start_time,
            "duration": duration,
            "original_filename": filename,
            "video_id": video_id
        }
        with open(meta_path, "w") as m:
            json.dump(metadata, m)

    return {"status": "chunk_received", "index": index, "video_id": video_id}


@router.post("/upload_chunk/finish")
async def finish_upload(video_id: str = Form(...)):
    file_dir = os.path.join(CHUNK_DIR, video_id)

    if not os.path.exists(file_dir):
        return JSONResponse({"error": "no_chunks_found"}, status_code=400)

    # Ordenar chunks
    parts = sorted(
        [p for p in os.listdir(file_dir) if p.endswith(".part")],
        key=lambda x: int(x.replace(".part", ""))
    )

    # Criar vídeo final
    final_filename = f"{video_id}.mp4"
    final_path = os.path.join(CHUNK_DIR, final_filename)

    with open(final_path, "wb") as final_file:
        for part in parts:
            part_path = os.path.join(file_dir, part)
            with open(part_path, "rb") as p:
                final_file.write(p.read())

    # Ler metadata
    meta_path = os.path.join(META_DIR, f"{video_id}.json")
    metadata = json.load(open(meta_path))

    # Upload do vídeo
    with open(final_path, "rb") as f:
        upload_bytes(f"incoming/{final_filename}", f.read(), "video/mp4")

    # Upload do JSON
    json_name = f"{video_id}.json"
    upload_bytes(
        f"incoming/{json_name}",
        json.dumps(metadata).encode("utf-8"),
        "application/json"
    )

    # Limpeza
    shutil.rmtree(file_dir)
    os.remove(final_path)
    os.remove(meta_path)

    return {"status": "upload_complete", "metadata": metadata}
