# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from game_manager import GameManager

app = FastAPI()
manager = GameManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"model_path": file_path}


@app.post("/start-game")
async def start_game(model_path: str):
    return manager.start_game(model_path)


@app.get("/state")
async def get_state():
    return manager.get_state()


@app.post("/human-move")
async def human_move(action_index: int):
    return manager.apply_human_move(action_index)

@app.post("/launch-gui")
async def launch_gui():
    return manager.launch_gui()

@app.get("/playable-actions")
async def playable_actions():
    return manager.get_playable_actions()

@app.post("/human-move")
async def human_move(action_index: int):
    return manager.apply_human_move(action_index)
