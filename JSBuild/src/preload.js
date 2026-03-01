// preload.js
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
    sendMessage: (msg) => ipcRenderer.send("send-to-python", msg),
    onResponse: (callback) =>
        ipcRenderer.on("python-response", (_, data) => callback(data)),
    onImageDone: (callback) =>
        ipcRenderer.on("image-done", callback)
});

window.addEventListener("DOMContentLoaded", () => {
  // nothing here yet
})
