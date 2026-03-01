const { app, BrowserWindow } = require('electron')

const createWindow = () => {
    const win = new BrowserWindow({
        width: 1080,
        height: 780,
    })

    win.loadFile('Cybel Dashboard.html')

}


app.whenReady().then(() => {
    createWindow()
})