document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("startGameBtn");
  const fileInput = document.querySelector(".file-input");
  const statusText = document.querySelector(".status p");
  const log = document.querySelector(".log");

  let modelPath = null;

  function addLog(message) {
    const entry = document.createElement("p");
    entry.textContent = message;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
  }

  async function uploadModel(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/upload-model", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    return data.model_path;
  }

  async function startGameBackend() {
    await fetch("http://localhost:8000/start-game?model_path=" + modelPath, {
      method: "POST"
    });
  }

  async function fetchState() {
    const res = await fetch("http://localhost:8000/state");
    return await res.json();
  }

  async function refreshGame() {
  const stateRes = await fetch("http://localhost:8000/state");
  const state = await stateRes.json();

  const actionRes = await fetch("http://localhost:8000/playable-actions");
  const actionData = await actionRes.json();

  renderBoard(state);
  renderActions(actionData.actions);
  }

  function renderBoard(state) {
    const boardDiv = document.querySelector(".board-placeholder");
    boardDiv.innerHTML = "<pre>" + JSON.stringify(state, null, 2) + "</pre>";
  }

  function renderActions(actions) {
    const log = document.querySelector(".log");
    log.innerHTML = "";

    actions.forEach((action, index) => {
      const btn = document.createElement("button");
      btn.textContent = action;
      btn.onclick = async () => {
        await fetch("http://localhost:8000/human-move?action_index=" + index, {
          method: "POST"
        });
        refreshGame();
      };
      log.appendChild(btn);
    });
  }


  startBtn.addEventListener("click", async () => {
    if (!fileInput.files[0]) {
      statusText.textContent = "Upload model first.";
      return;
    }

    statusText.textContent = "Uploading model...";
    modelPath = await uploadModel(fileInput.files[0]);

    statusText.textContent = "Starting game...";
    await startGameBackend();

    statusText.textContent = "Game running.";
    addLog("Game started.");
  });
});
