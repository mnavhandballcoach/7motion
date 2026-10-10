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
    """
    Recebe um chunk + metadata e guarda tudo.
    """

    # Pasta dos chunks
    file_dir = os.path.join(CHUNK_DIR, filename)
    os.makedirs(file_dir, exist_ok=True)

    # Guardar chunk
    part_path = os.path.join(file_dir, f"{index}.part")
    with open(part_path, "wb") as f:
        f.write(await chunk.read())

    # Guardar metadata (apenas 1 vez)
    meta_path = os.path.join(META_DIR, f"{filename}.json")
    if not os.path.exists(meta_path):
        metadata = {
            "email": email,
            "name": name,
            "start": start_time,
            "duration": duration
        }
        with open(meta_path, "w") as m:
            json.dump(metadata, m)

    return {"status": "chunk_received", "index": index}


@router.get("/upload_chunk")
async def finish_upload(filename: str, finish: int = 0):
    """
    Junta todos os chunks, cria o vídeo final, envia vídeo + JSON para o B2.
    """

    if finish != 1:
        return JSONResponse({"error": "invalid_finish_flag"}, status_code=400)

    file_dir = os.path.join(CHUNK_DIR, filename)

    if not os.path.exists(file_dir):
        return JSONResponse({"error": "no_chunks_found"}, status_code=400)

    # Ordenar chunks
    parts = sorted(
        [p for p in os.listdir(file_dir) if p.endswith(".part")],
        key=lambda x: int(x.replace(".part", ""))
    )

    # Criar vídeo final
    final_path = os.path.join(CHUNK_DIR, f"final_{filename}")
    with open(final_path, "wb") as final_file:
        for part in parts:
            part_path = os.path.join(file_dir, part)
            with open(part_path, "rb") as p:
                final_file.write(p.read())

    # Ler metadata
    meta_path = os.path.join(META_DIR, f"{filename}.json")
    metadata = json.load(open(meta_path))

    # Enviar vídeo para B2
    with open(final_path, "rb") as f:
        upload_bytes(f"incoming/{filename}", f.read(), "video/mp4")

    # Enviar JSON para B2
    json_name = filename.replace(".mp4", ".json")
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
