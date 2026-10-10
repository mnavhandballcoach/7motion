from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os

from api.upload_chunk import router as upload_router
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.upload_chunk import router as upload_router

app = FastAPI()

# Servir ficheiros estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Rota principal opcional
@app.get("/")
def root():
    return {"status": "ok", "message": "7motion backend is running"}

# Router do upload
app.include_router(upload_router)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(static_dir, "upload.html"), "r").read()

app.include_router(upload_router)
