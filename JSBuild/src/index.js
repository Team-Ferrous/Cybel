const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const fs = require("fs");
const decoder        = require('./decoder');
const instanceEngine = require('./instance_engine');
const path = require("path");

const { 
    initialize, 
    sendMessage, 
    setGroqKey, 
    setGenerationMode, 
    setTemperature, 
    setContextWindowKey, 
    updateCharacter,
    saveDocument,
    loadDocument,
    deleteDocument,
    replicateDocument,
    mergeDocument,
    exportDocument,
    setEngine
} = require('./backend');

let mainWindow;
let vectorStore;

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
});


ipcMain.handle('open-file-dialog', async () => {
    const result = await dialog.showOpenDialog({
        properties: ['openFile', 'multiSelections']
    });
    
    return result.filePaths;
});

ipcMain.handle('save-file-dialog', async () => {
    const result = await dialog.showSaveDialog({
        properties: ['saveFile', 'multiSelections']
    });

    return result.filePaths;
});

ipcMain.handle("engine:update", async (_, newConfig) => {
    setEngine(newConfig);
    console.log("Engine config updated:", newConfig);
});

ipcMain.handle("chat:setGroqKey", async (_, key) => {
    setGroqKey(key);
    return true;
});

ipcMain.handle("chat:setMode", async (_, mode) => {
    setGenerationMode(mode);
    return true;
});

ipcMain.handle("chat:setTemperature", async (_, key) => {
    setTemperature(key);
    return true;
});

ipcMain.handle("engine:save_document", async (event, doc) => {
  return await saveDocument(doc);
});

ipcMain.handle("engine:load_document", async (event, doc) => {
  return await loadDocument(doc);
});

ipcMain.handle("engine:delete_document", async (event, doc) => {
  return await deleteDocument(doc);
});

ipcMain.handle("engine:replicate_document", async (event, doc) => {
  return await replicateDocument(doc);
});

ipcMain.handle("engine:merge_document", async (event, doc) => {
  return await mergeDocument(doc);
});

ipcMain.handle("engine:export_document", async (event, doc) => {
  return await exportDocument(doc);
});

ipcMain.handle("engine:spawn_instance", async (event, config) => {
  return await instanceEngine.spawn(config);
});

ipcMain.handle("chat:setContextWindowKey", async (_, mode) => {
    setContextWindowKey(mode);
    return true;
});

ipcMain.handle("engine:updateCharacter", async (_, mode) => {
    updateCharacter(mode);
    return true;
});

ipcMain.handle('decode-directory', async (event, args) => {
  return await decoder.decodeDirectory(args.path, args.options);
});

ipcMain.handle("rag:ingest", async (event, paths) => {
    for (const file of paths) {
        const raw = fs.readFileSync(file, "utf8");
        const chunks = chunkText(raw);
        const embeddings = await embed(chunks);
        vectorStore = instanceEngine("instanceId", myQueryVector, 10); 
        await vectorStore.add(embeddings);
    }
    return { success: true };
});

ipcMain.handle("rag:query", async (event, qry) => {
    vectorStore = instanceEngine("instanceId", myQueryVector, 10); 
    await vectorStore.query(qry);
    return { success: true };
});

ipcMain.handle("rag:clear", async (event, idx) => {
    vectorStore = instanceEngine("instanceId", myQueryVector, 10); 
    await vectorStore[idx].clear();
    return { success: true };
});

ipcMain.handle("engine:ingest_documents", async (event, { instanceId, files }) => {
  if (!files || files.length === 0) return { success: false, message: "No files selected" };

  try {
    const allChunks = [];

    for (const file of files) {
      const raw = fs.readFileSync(file, "utf8");

      // Decode chat or raw text
      const chunks = decoder.decodeChat(raw); // returns array of strings
      allChunks.push(...chunks);
    }

    const result = await instanceEngine.ingestDocuments(instanceId, allChunks);

    return { success: true, ingested: result.count };
  } catch (err) {
    console.error("Error ingesting documents:", err);
    return { success: false, message: err.message };
  }
});

// -------------CHAT SPECIFIC FEATURES--------------
ipcMain.handle("chat:send", async (_, userInput) => {
    //console.log("backend:", require("./backend"));
    const result = await sendMessage(userInput);
    return result;
});