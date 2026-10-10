from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.upload_chunk import router as upload_router

app = FastAPI()

# CORS para permitir chamadas do GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminho correto para a pasta static dentro do container
static_dir = "/app/static"

# Servir ficheiros estáticos
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Página inicial → devolve o upload.html
@app.get("/", response_class=HTMLResponse)
def index():
    return open("/app/static/upload.html", "r").read()

# Router do upload
app.include_router(upload_router)
