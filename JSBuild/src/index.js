const { app, BrowserWindow, ipcMain } = require("electron");
const { initialize, sendMessage }     = require("./backend.js");
const path = require("path");
//import { app, BrowserWindow, ipcMain } from "electron";
//import { initialize, sendMessage } from "./backend.js";

let mainWindow;

async function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1080,
        height: 720,
        webPreferences: {
            preload: path.join(__dirname, "preload.js"),
            contextIsolation: true
        }
    });
    mainWindow.loadFile("index.html");
}

app.whenReady().then(async () => {

    await initialize();   // build/load vector index FIRST
    await createWindow(); // then show UI

    ipcMain.handle("chat-message", async (_, message) => {
        return await sendMessage(message);
    });
});