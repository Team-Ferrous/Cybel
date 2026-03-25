import fs from "fs";
import { Decoder }        from './decoder.js';
import { Worker }         from 'worker_threads'; // <-- important!
import { app, BrowserWindow, ipcMain } from "electron";
import { 
    sendMessage, 
    setTokenKey, 
    setTemperature, 
    setContextWindowKey, 
    updateCharacter,
    saveDocument,
    loadDocument,
    deleteDocument,
    replicateDocument,
    mergeDocument,
    exportDocument,
    setEngine,
    getEngineInstance,
    setGenerationMode,
    ingestSpreadsheetToFAISS,
    ingestSheets,
    authorizeSheets,
    embeddings,
    makeQRCode
} from './backend.js';

import { spawn } from "child_process";
import path      from "path";
import { dirname }       from 'node:path';
import { fileURLToPath } from 'node:url';
import { google } from 'googleapis';
import { createRequire } from 'node:module';
const require  = createRequire(import.meta.url);
const os       = require("os");
const crypto = require("crypto");

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);

const base = os.hostname() + os.arch() + crypto.randomUUID();
const userId = crypto.createHash("sha256").update(base).digest("hex");

let pythonServer;
//import { InstanceEngine } from './instance_engine'

async function trackEvent(name, payload = {}) {
    await fetch("https://your-server.com/analytics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event: name, ...payload })
    });
}

async function openGDDialog() {
    const auth = await authorizeSheets(); // reuse your OAuth helper
    const drive = google.drive({ version: 'v3', auth });

    // List files in a folder (optional: folderId for CybelIngest)
    const res = await drive.files.list({
        q: `'${folderId || "root"}' in parents and mimeType='application/vnd.google-apps.spreadsheet'`,
        fields: 'files(id, name)',
    });

    const files = res.data.files || [];
    return new Promise(resolve => {
        // Open an Electron modal with checkboxes for files
        const win = new BrowserWindow({
            width: 600,
            height: 400,
            modal: true,
            webPreferences: { nodeIntegration: true, contextIsolation: false }
        });

        win.loadFile('gdrive_dialog.html'); // a simple HTML + JS for selecting files

        // Send file list to window
        win.webContents.once('did-finish-load', () => {
            win.webContents.send('gdrive-file-list', files);
        });

        // Listen for selection
        ipcMain.once('gdrive-file-selected', async (_, selectedFiles) => {
            win.close();
            resolve(selectedFiles); // array of { id, name }
        });
    });
}

//specifically and ONLY for SPARC3D-SDF and 3d Asset Gen models in Python
function startPythonServer() {
    const script = path.join(__dirname, "python", "sparc_server.py");
    pythonServer = spawn("python", [script]);
    pythonServer.stdout.on("data", (data) => {
        console.log(`PYTHON: ${data}`);
    });

    pythonServer.stderr.on("data", (data) => {
        console.error(`PYTHON ERROR: ${data}`);
    });

    pythonServer.on("close", (code) => {
        console.log(`Python server exited with code ${code}`);
    });
}

let mainWindow;
let vectorStore;
async function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1080,
        height: 720,
        webPreferences: {
            preload: path.join(__dirname, "preload.js"),
            contextIsolation: true,
            nodeIntegration: false
        }
    });
    mainWindow.loadFile("index.html");
}

function initializeWorker() {
    return new Promise((resolve, reject) => {
        const worker = new Worker('./initialize.js');
        worker.on('message', resolve);
        worker.on('error', reject);
        worker.on('exit', (code) => {
            if (code !== 0) reject(new Error(`Worker stopped with code ${code}`));
        });
    });
}


ipcMain.handle("engine:qrcode", async (event) => {
        return makeQRCode("https://google.com");
});


ipcMain.handle("analytics:track", async (_, name, payload) => {
    await trackEvent(name, payload);
}); 

ipcMain.handle("engine:update", async (_, newConfig) => {
    setEngine(newConfig);
    trackEvent("feature_used", { userId, feature: "engine:update" });
    console.log("Engine config updated:", newConfig);
});

ipcMain.handle("chat:setTokenKey", async (_, key) => {
    setTokenKey(key);
    return true;
});

ipcMain.handle("engine:set-generation-mode", async (_, mode) => {
    trackEvent("feature_used", { userId, feature: "engine:set-generation-mode" });
    setGenerationMode(mode);
    return true;
});

ipcMain.handle("engine:loadModel", async (_, mName) => {
    loadModel(mName)
    return true;
});

ipcMain.handle("chat:setTemperature", async (_, key) => {
    setTemperature(key);
    trackEvent("feature_used", { userId, feature: "chat:setTemperature" });
    return true;
});

ipcMain.handle("engine:save_document", async (event, doc) => {
    trackEvent("feature_used", { userId, feature: "engine:save_document" });
    return await saveDocument(doc);
});

ipcMain.handle("engine:load_document", async () => {
    trackEvent("feature_used", { userId, feature: "engine:load_document" });
    return await loadDocument();
});

ipcMain.handle("engine:delete_document", async (event, doc) => {
    trackEvent("feature_used", { userId, feature: "engine:delete_document" });
    return await deleteDocument(doc);
});

ipcMain.handle("engine:replicate_document", async (event, doc) => {
    trackEvent("feature_used", { userId, feature: "engine:replicate_document" });
    return await replicateDocument(doc);
});

ipcMain.handle("engine:merge_document", async (event, doc) => {
    trackEvent("feature_used", { userId, feature: "engine:export_document" });
    return await mergeDocument(doc);
});

ipcMain.handle("engine:export_document", async (event, doc) => {
    trackEvent("feature_used", { userId, feature: "engine:export_document" });
    return await exportDocument(doc);
});

ipcMain.handle("engine:spawn_instance", async (event, config) => {
    trackEvent("feature_used", { userId, feature: "engine:export_document" });
    return await getEngineInstance().spawn(config);
});

ipcMain.handle("engine:save_config", async (event, doc) => {
  return await getEngineInstance().saveConfig(doc);
});

ipcMain.handle("engine:load_config", async () => {
  return await getEngineInstance().loadConfig();
});

ipcMain.handle("engine:get_config", async () => {
  return await getEngineInstance().config;
});

ipcMain.handle("chat:setContextWindowKey", async (_, mode) => {
    setContextWindowKey(mode);
    return true;
});

ipcMain.handle("engine:updateCharacter", async (_, mode) => {
    trackEvent("feature_used", { userId, feature: "engine:updateCharacter" });
    updateCharacter(mode);
    return true;
});

ipcMain.handle('decode-directory', async (event, args) => {
    trackEvent("feature_used", { userId, feature: "rag:ingest" });
    return await Decoder.decodeDirectory(args.path, args.options);
});

ipcMain.handle("rag:ingest", async (event, paths) => {
    for (const file of paths) {
        const raw = fs.readFileSync(file, "utf8");
        const chunks = chunkText(raw);
        const emb = await embedder(chunks, { pooling: "mean", normalize: true });

        embeddings.push(new Float32Array(emb.data));

        let eng = getEngineInstance();
        vectorStore = eng("instanceId", myQueryVector, 10); 
        await vectorStore.add(embeddings);
        trackEvent("feature_used", { userId, feature: "rag:ingest" });
    }
    return { success: true };
});

ipcMain.handle("rag:query", async (event, qry) => {
    let eng = getEngineInstance();
    vectorStore = eng("instanceId", myQueryVector, 10); 
    trackEvent("feature_used", { userId, feature: "rag:query" });
    await vectorStore.query(qry);
    return { success: true };
});

ipcMain.handle("rag:clear", async (event, idx) => {
    let eng = getEngineInstance();
    vectorStore = eng("instanceId", myQueryVector, 10); 
    trackEvent("feature_used", { userId, feature: "rag:clear" });

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
      const chunks = Decoder.decodeChat(raw); // returns array of strings
      allChunks.push(...chunks);
    }
    let eng = getEngineInstance();
    const result = await eng.ingestDocuments(instanceId, allChunks);
    trackEvent("feature_used", { userId, feature: "engine:ingest_documents" });

    return { success: true, ingested: result.count };
  } catch (err) {
    console.error("Error ingesting documents:", err);
    return { success: false, message: err.message };
  }
});

ipcMain.handle("engine:getEngineInstance", () => {
    let eng = getEngineInstance();
    trackEvent("feature_used", { userId, feature: "engine:getEngineInstance" });

    return eng ? eng : { success: false, message: "No engine instance available" };
});

ipcMain.handle("engine:spawnAgent", async (event, config) => {
    let eng = getEngineInstance();
    trackEvent("feature_used", { userId, feature: "engine:spawnAgent" });
    return await eng.spawnAgent(config);
});

ipcMain.handle("engine:deleteAgent", async (_, config) => {
    let eng = getEngineInstance();
    return await eng.destroy(config.id);
});

ipcMain.handle("get-local-models", async () => {
    const modelsDir = path.join(__dirname, "models"); // adjust path
    if (!fs.existsSync(modelsDir)) return [];

    const files = fs.readdirSync(modelsDir);
    trackEvent("feature_used", { userId, feature: "engine:get-local-models" });
    // assume each model has a folder named after it
    return files.filter(f => fs.statSync(path.join(modelsDir, f)).isDirectory());
});

// -------------CHAT SPECIFIC FEATURES--------------
ipcMain.handle("chat:send", async (_, userInput) => {
    //console.log("backend:", require("./backend"));
    const result = await sendMessage(userInput);
    trackEvent("feature_used", { userId, feature: "chat:send" });
    return result;
});

ipcMain.handle("chat:setMode", (event, mode) => {
  console.log("mode set to", mode)
    trackEvent("feature_used", { userId, feature: "chat:setMode" });
  return `Mode changed to ${mode}`
})

ipcMain.handle("spawn-agent", async (event, config) => {
    let eng = getEngineInstance();
    trackEvent("feature_used", { userId, feature: "spawn-agent" });

    return await eng.spawn(config);
});


ipcMain.handle("get-agent", (event, id) => {
    let eng = getEngineInstance();
    trackEvent("feature_used", { userId, feature: "get-agent" });

    return eng.get(id);
});

// RAG Ingestion and Querying
ipcMain.handle("ingest-google-sheet", async (_, spreadsheetId) => {
    try {
        trackEvent("feature_used", { userId, feature: "ingest-google-sheet" });
        return await ingestSpreadsheetToFAISS(spreadsheetId);
    } catch (err) {
        return { success: false, error: err.message };
    }
});

ipcMain.handle("open-gdrive-dialog", async () => {
    try {
        trackEvent("feature_used", { userId, feature: "ingest-google-sheet" });
        const selectedFiles = await openGDDialog().then(result => {
            if (!result.success) {
                alert("Failed to open Google Drive dialog: " + result.error);
            } else {
                console.log("Drive files selected:", result.instance);
            }
        })
        .catch(err => {
            console.error("Agent spawn error:", err);
        }); // [{id, name}, ...]
        const results = [];

        for (const file of selectedFiles) {
            await ingestSheets(null, file);
            results.push({ name: file.name, rowsIngested: rows.length });
        }

        return { success: true, details: results };
    } catch (err) {
        return { success: false, error: err.message };
    }
});

// In main
app.whenReady().then(async () => {
    await createWindow();
    startPythonServer();
    initializeWorker().then((embeddingIndex) => {
        console.log("FAISS loaded in worker!");
    });
    trackEvent("first_action_completed", { userId });
});