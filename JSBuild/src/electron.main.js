import { app, BrowserWindow } from "electron"
import path from "path"

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1024,
    height: 768,
    icon: path.join(__dirname, 'assets', 'cybel.ico'), // <-- important
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  // Load your app — change to your actual local dev URL or build index.html
  mainWindow.loadURL("http://localhost:3000")
}

app.whenReady().then(createWindow)

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit()
})
