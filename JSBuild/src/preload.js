// preload.js
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
    sendMessage: (msg)      => ipcRenderer.send("chat-message", msg),
    //onResponse: (callback)  => ipcRenderer.on("chat-response", (_, data) => callback(data)),
    onImageDone: (callback) => ipcRenderer.on("image-done", callback),
    onAssetDone: (callback) => ipcRenderer.on("asset-done", callback),
    loadModel: (path)       => ipcRenderer.invoke("load-model", path),
    onToken: (callback) => ipcRenderer.on("stream-token", (_, t) => callback(t))
});

window.addEventListener("DOMContentLoaded", () => {
  // nothing here yet
})
async function loadSelectedModel() {
    const path = document.getElementById("model-path").value;
    if (!path) {
        showModal("Model Error", "No model path specified.");
        return;
    }

    showModal("Engine", "Loading model...");

    try {
        await window.api.loadModel(path);
        showModal("Engine Online", "Model successfully loaded.");
    } catch (err) {
        showModal("Load Failed", err.message);
    }
}

