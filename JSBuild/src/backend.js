// backend.js
import fs     from "fs";
import crypto from "crypto";
import Store  from 'electron-store';
import path   from 'path';
import { dialog, ipcMain, shell, BrowserWindow }   from "electron";

import OpenAI             from "openai";
import { google       }   from "googleapis";
import { createXai    }   from '@ai-sdk/xai';
import { generateText }   from 'ai';
import { error        }   from "node:console";
import { pipeline, env     }   from "@huggingface/transformers";
import { spawn        }   from "child_process";

import { InstanceEngine } from "./instance_engine.js";
import { embeddingDim,  embeddingIndex, embedText   } from "./embeddings.js";
import { findGenerator, modelRegistry,  activeStack } from './model_switcher.js'

import { dirname       } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { reflect, RecallMemory, RetainMemory } from "./hindsight.js";
import { Decoder }        from './decoder.js';
import dotenv from "dotenv";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);
const require    = createRequire(import.meta.url);
const { IndexFlatL2 }  = require(path.resolve(__dirname, './node_modules/faiss-node/build/Release/faiss-node'));
dotenv.config({ path: path.join(__dirname, ".env") });

const engine            = new InstanceEngine();
const SHEETS_TOKEN_PATH = path.join(__dirname, "sheets_token.json");
const SHEETS_CREDENTIALS_PATH = path.join(__dirname, "sheets_credentials.json"); // from GCP
let sheetsAuth;
// main.js
const store = new Store();

let CONFIG = {
    temperature:      0.75,
    contextWindow:    4096,
    generationMode:   "groq", //  "groq" | "local" | "grok" | "verso"
    model:            "openai/gpt-oss-20b",
    citation_options: 'enabled',
    tokenKey:          store.get("tokenKey") || null, //localStorage.getItem("groqKey") //process.env.GROQ_API_KEY,
    agent_instances:   engine.instances,
};

async function makeQRCode(){
    // Generate QR code as a Base64 data URL
    const require = createRequire(import.meta.url);
    const QRCode = require("qrcode");
    const url = "https://google.com"; // or userId
    return await QRCode.toDataURL(url); // returns "data:image/png;base64,..."
}

async function requestOAuthCode() {
    return new Promise(resolve => {
        const authWin = new BrowserWindow({
            width: 400,
            height: 220,
            modal: true,
            title: "Google Authorization",
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: false
            }
        });

        authWin.loadFile("./oauth_code.html");
        ipcMain.once("google-sheets-code", (_, code) => {
            authWin.close();
            resolve(code);
        });
    });
}
async function authorizeSheets() {
    if (sheetsAuth) return sheetsAuth;
    const credentialsRaw = await fs.promises.readFile(SHEETS_CREDENTIALS_PATH, "utf-8");
    const credentials = JSON.parse(credentialsRaw);
    const { client_secret, client_id, redirect_uris } = credentials.installed;
    const oAuth2Client = new google.auth.OAuth2(
        client_id,
        client_secret,
        redirect_uris[0]
    );

    console.log("Authorizing Google Sheets access...");

    try {
        const tokenRaw = await fs.promises.readFile(SHEETS_TOKEN_PATH, "utf-8");
        const token = JSON.parse(tokenRaw);

        oAuth2Client.setCredentials(token);

        console.log("Using existing Google Sheets token");

    } catch (err) {

        console.log("No token found, starting OAuth flow");

        const authUrl = oAuth2Client.generateAuthUrl({
            access_type: "offline",
            scope: ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        });

        await shell.openExternal(authUrl);
        const code = await requestOAuthCode();
        const { tokens } = await oAuth2Client.getToken(code);
        oAuth2Client.setCredentials(tokens);
        await fs.promises.writeFile(
            SHEETS_TOKEN_PATH,
            JSON.stringify(tokens),
            "utf-8"
        );
    }

    console.log("Google Sheets authorization successful");

    sheetsAuth = oAuth2Client;
    return oAuth2Client;
}

async function fetchSpreadsheet(spreadsheetId, range = "Sheet1!A:F") {
    const auth = await authorizeSheets();
    const sheets = google.sheets({ version: "v4", auth });
    const res = await sheets.spreadsheets.values.get({ spreadsheetId, range });
    const rows = res.data.values || [];

    if (!rows.length) return [];

    const header = rows.shift();
    const formattedRows = rows.map(row => {
        const obj = {};
        header.forEach((key, i) => { obj[key] = row[i] ?? ""; });
        return obj;
    });

    return formattedRows;
}

async function ingestSpreadsheetToFAISS(spreadsheetId, range = "Sheet1!A:F") {
    const rows = await fetchSpreadsheet(spreadsheetId, range);
    if (!rows.length) return { success: false, message: "No data found in sheet." };

    await ingestSheets(null, rows);
    return { success: true, rowsIngested: rows.length };
}

function createGroqClient() {
    return new Groq({ apiKey: CONFIG.groqKey });
}

//const DOCUMENT_DIR = path.join(app.getPath('userData'), 'documents');
const DOCUMENT_DIR = path.join(__dirname, "documents");

let embedder;
let generator;
let docs       = []
let embeddings = [];

// Usage:
//const vector = await getOAIEmbedding("Hello world");
//console.log(vector.length); // 1536 for text-embedding-3-small
//Example using HuggingFace transformers.js:
async function getOAIEmbedding(text) {
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    async function embedder(text, options = {}) {
        const response = await client.embeddings.create({
            model: "text-embedding-3-small",
            input: text
        });
        return response.data[0].embedding;
    }
    
    const vector = await embedder(text);

    return vector;
}

async function getHFEmbedding(text) {
    const embedPipeline = await pipeline("feature-extraction", "sentence-transformers/all-MiniLM-L6-v2");

    async function embedder(txt) {
        const result = await embedPipeline(txt);
        // result is a nested array, flatten if needed
        return result[0].flat();
    }

    const vector = await embedder(text);
    return vector;
}

// Active conversation stack
let activeConversationStack = []; // each entry: { user: string, bot: string }

function pushToConversationStack(userInput, botResponse) {
    activeConversationStack.push({ user: userInput, bot: botResponse });
}

function getConversationHistory() {
    return activeConversationStack.map(entry => `User: ${entry.user}\nBot: ${entry.bot}`).join("\n\n");
}

// ---------------------------
// Paths
// ---------------------------
const inputDir   = path.join(__dirname, "Input_JSON");
const logsDir    = path.join(__dirname, "Logs");
const indexPath  = path.join(__dirname, "vector.index");
const metaPath   = path.join(__dirname, "vector_docs.json");
const hashPath   = path.join(__dirname, "doc_hash.txt");

if (!fs.existsSync(inputDir)) fs.mkdirSync(inputDir, { recursive: true });
if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });

//----------------------------
// Utility & Engine Functions
//----------------------------
function setGenerationMode(gen){
    CONFIG.generationMode = gen;
}

function setTemperature(gen){
    CONFIG.temperature = gen;
}

function setContextWindowKey(key){
    CONFIG.contextWindow = key;
}

function updateEngine(engine){
    CONFIG.model = engine;
}

function setTokenKey(key){
    CONFIG.tokenKey = key;
    store.set("tokenKey", key);
}

function setEngineInstance(eng){
    engine = eng;
}

function setEngine(conf){
    CONFIG.engine = conf;
    store.set("current-engine", CONFIG.engine)
}

function getEngine(){
    return store.get("current-engine");
}

function getEngineInstance(){
    return engine;
}

async function ingestSheets(id, rows) {
    if (!embeddingIndex) embeddingIndex = new IndexFlatL2({ dims: embeddingDim });
    for (const row of rows) {
        const text = Object.entries(row)
                           .map(([k,v]) => `${k}: ${v}`)
                           .join(", ");
        embedder  = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { token: config.hfToken });
        const emb = await embedder(text, { pooling: "mean", normalize: true });
        embeddings.push(Array.from(emb.data));
        docs.push({ content: text, metadata: row });
    }

    await embeddingIndex.add(embeddings);
    // Optional: persist for cache
    fs.writeFileSync(metaPath,  JSON.stringify(docs), "utf-8");
    fs.writeFileSync(indexPath, JSON.stringify(embeddings), "utf-8");
}

function chunkText(text, chunkSize = 500, overlap = 50) {
    const words = text.split(/\s+/); // split by whitespace
    const chunks = [];

    for (let i = 0; i < words.length; i += chunkSize - overlap) {
        const chunk = words.slice(i, i + chunkSize).join(" ");
        chunks.push(chunk);
    }

    return chunks;
}

async function saveDocument(doc) {
  try {
    const result = await dialog.showSaveDialog({
      title: "Save Document",
      defaultPath: doc.title + ".json",
      filters: [
        { name: "JSON Files", extensions: ["json"] }
      ]
    });

    if (result.canceled) {
      return { success: false, canceled: true };
    }

    const filePath = result.filePath;
    await fs.promises.writeFile(
      filePath,
      JSON.stringify(doc, null, 2),
      "utf-8"
    );

    return { success: true, path: filePath };
  } catch (err) {
    console.error("Save error:", err);
    return { success: false, error: err.message };
  }
}
async function loadDocument() {
  try {
        const result = await dialog.showOpenDialog({
            title: "Import Document",
            defaultPath: "default.txt",
            filters: [
              {name: "Text or JSON", extensions: ["txt", "json" ]},
              {name: "Spreadsheet Files", extensions: [ "xlsx", "xls", "csv", "tsv"]},  
              {name: "PDF Files extensions:", extensions: [ "pdf"]}
            ]
        });

        if (result.canceled) {
            return { success: false, canceled: true };
        }

        const importPath = result.filePaths[0];
        const data       = await fs.readFile(importPath , "utf-8");
        if (importPath.endsWith(".pdf")) {
            // embed in faiss
            embeddingIndex = new IndexFlatL2({ dims: embeddingDim });
            //await Promise.all(docs.map(embedDoc))
            const chunks = chunkText(data);
            for (const doc of chunks) {
                // Skip anything that isn’t an object or missing content
                if (doc && typeof doc.content === "string" && doc.content.trim() !== "") {
                    const emb = await embedder(doc.content, { pooling: "mean", normalize: true });
                    embeddings.push(Array.from(emb.data));
                } else {
                    console.warn("⚠️ Skipping doc with invalid content:", doc);
                }
            }
            await embeddingIndex.add(embeddings);
            console.log(`⚡ Built new FAISS index, ntotal: ${embeddingIndex.ntotal()}`);
            // return text content for now, but ideally we’d want to keep the PDF structure and metadata
            return { success: true, path: importPath, document: data };
        }
        else if (importPath.endsWith(".xlsx") || importPath.endsWith(".xls") || importPath.endsWith(".csv") || importPath.endsWith(".tsv")) {
            // embed in faiss
            embeddingIndex = new IndexFlatL2({ dims: embeddingDim });
            //await Promise.all(docs.map(embedDoc))
            const chunks = chunkText(data);
            for (const doc of chunks) {
                // Skip anything that isn’t an object or missing content
                if (doc && typeof doc.content === "string" && doc.content.trim() !== "") {
                    ingestSheets(null, [doc]); // also add to sheets for agent use
                } else {
                    console.warn("⚠️ Skipping doc with invalid content:", doc);
                }
            }
            await embeddingIndex.add(embeddings);
            console.log(`⚡ Built new FAISS index, ntotal: ${embeddingIndex.ntotal()}`);
            // return text content for now, but ideally we’d want to keep the PDF structure and metadata
            return { success: true, path: importPath, document: data };
        }
        else if (importPath.endsWith(".txt")) {
            return { success: true, path: importPath, document: data };
        }
        else if (importPath.endsWith(".json")) {
            return { success: true, path: importPath, document: JSON.parse(data) };
        }
    } catch (err) {
        return { success: false, error: err.message };
    }
}

async function deleteDocument(doc) {
  try {
    const result = await dialog.showMessageBox({
            title: "Select Document to Delete",
            defaultPath: doc.title + ".json",
            filters: [
            { name: "JSON Files", extensions: ["json"] }
            ]
        });

        if (result.canceled) {
            return { success: false, canceled: true };
        }

        const importPath = result.filePath;
        await fs.unlink(importPath);
        return { success: true, path: importPath };
    } catch (err) {
        return { success: false, error: err.message };
    }
}

async function replicateDocument(doc) {
  try {
    const sourcePath = path.join(DOCUMENT_DIR, doc.title + ".json");
    const targetPath = path.join(DOCUMENT_DIR, doc.newTitle + ".json");

    const result = await dialog.showOpenDialog({
            title: "Import Document",
            defaultPath: doc.title + ".json",
            filters: [
            { name: "JSON Files", extensions: ["json"] }
            ]
        });

        if (result.canceled) {
            return { success: false, canceled: true };
        }
        const importPath = result.filePath;
        const data = await fs.readFile(sourcePath, "utf-8");
        const parsed = JSON.parse(data);

        parsed.title = doc.newTitle;
        await fs.writeFile(targetPath, JSON.stringify(parsed, null, 2), "utf-8");
        return { success: true, path: importPath };
    } catch (err) {
        return { success: false, error: err.message };
    }
    }

async function mergeDocument(doc) {
  try {
    const basePath = path.join(DOCUMENT_DIR, doc.baseTitle + ".json");
    const mergePath = path.join(DOCUMENT_DIR, doc.mergeTitle + ".json");
    const outputPath = path.join(DOCUMENT_DIR, doc.outputTitle + ".json");

    const base = JSON.parse(await fs.readFile(basePath, "utf-8"));
    const merge = JSON.parse(await fs.readFile(mergePath, "utf-8"));

    const merged = {
      title: doc.outputTitle,
      content: base.content + "\n\n" + merge.content
    };

    await fs.writeFile(outputPath, JSON.stringify(merged, null, 2), "utf-8");

    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function exportDocument(doc) {
  try {
    const sourcePath = path.join(DOCUMENT_DIR, doc.title + ".json");
    const exportPath = path.join(DOCUMENT_DIR, doc.title + ".txt");

    const data = JSON.parse(await fs.readFile(sourcePath, "utf-8"));

    await fs.writeFile(exportPath, data.content, "utf-8");

    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function updateCharacter(doc) {
  try {
    const filePath = path.join(DOCUMENT_DIR, doc.title + ".json");
    const data = JSON.parse(await fs.readFile(filePath, "utf-8"));

    data.character = doc.character;
    await fs.writeFile(filePath, JSON.stringify(data, null, 2), "utf-8");
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

// ---------------------------
// Hash Helper
// ---------------------------
function computeDocumentsHash() {
    const hasher = crypto.createHash("sha256");
    const files = fs.readdirSync(inputDir).filter(f => f.endsWith(".json")).sort();
    for (const file of files) {
        hasher.update(fs.readFileSync(path.join(inputDir, file)));
    }
    return hasher.digest("hex");
}

// ---------------------------
// Load Models
// ---------------------------
async function loadModels() {
    try {
        let hfToken = localStorage.getItem("hfToken");
        let msToken = localStorage.getItem("msToken");
        let groqToken = localStorage.getItem("groqToken");
        let grokToken = localStorage.getItem("grokToken");
        let HNToken   = localStorage.getItem("HNToken");

        console.log("Loading embedding model...");
        embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { token: hfToken });
        console.log("Embedding model loaded!");

        if (CONFIG.generationMode === "groq") {
            console.log("Loading Groq (requested) model...");
            generator = await pipeline("text-generation", "Xenova/phi-2", { token: groqToken });
            console.log("Generator model loaded!");
        }
        if (CONFIG.generationMode === "grok") {
            console.log("Loading Grok model...");
            generator = await pipeline("text-generation", "Xenova/phi-2", { token: grokToken });
            console.log("Generator model loaded!");
        }
        if (CONFIG.generationMode === "local") {
            let hfToggle = document.getElementById("hfBox");
            let msToggle = document.getElementById("msBox");

            if(hfToggle && hfToggle.checked && !msToggle.checked ){
                console.log("Loading local text-generation model...");
                generator = await pipeline("text-generation", "Xenova/phi-2", { token: hfToken });
                console.log("Generator model loaded!");
            }else if ( msToggle && msToggle.checked && !hfToggle.checked){
                console.log("Loading local text-generation model...");
                generator = await pipeline("text-generation", "Xenova/phi-2", { token: msToken });
                console.log("Generator model loaded!");        
            } else {
                return { success: false, error: "Please select exactly one model source (HuggingFace or ModelScope) in the settings." };
            }
        }
        if (CONFIG.generationMode === "verso") {
            console.log("Loading Verso model...");
            generator = await pipeline("text-generation", "Xenova/phi-2", { token: HNToken });
            console.log("Generator model loaded!");
        }
    } catch (err) {
        console.error("❌ Failed to load models:", err.message); // optional: show an Electron dialog or exit gracefully
    }
}

function getAllDocsFromInputJSON(inputFolder) {
    const files = fs.readdirSync(inputFolder).filter(f => f.endsWith(".json"));
    const docs = [];
    for (const file of files) {
        const filePath = path.join(inputFolder, file);
        const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));

        // Assume each JSON is an array of objects or a single object
        if (Array.isArray(data)) {
            docs.push(...data);
        } else {
            docs.push(data);
        }
    }

    return docs;
}

async function initialize() {
    try{
    await loadModels(); // load embedder & generator

    const currentHash = computeDocumentsHash();
    const hasCache = fs.existsSync(indexPath) &&
                     fs.existsSync(metaPath) &&
                     fs.existsSync(hashPath);

    if (hasCache) {
        const savedHash = fs.readFileSync(hashPath, "utf-8");
        docs = JSON.parse(fs.readFileSync(metaPath, "utf-8"));

        if (savedHash === currentHash) {
            // Load cached embeddings
            const cachedEmbeddings = JSON.parse(fs.readFileSync(indexPath, "utf-8")); 
            embeddingIndex = new IndexFlatL2({ dims: embeddingDim });
            if (embeddings.length === 0) {
                console.warn("No embeddings generated. Index will be empty.");
            }
            await embeddingIndex.add(cachedEmbeddings) //.add(cachedEmbeddings);
            embeddings.push(new Float32Array(emb.data));
            console.log("⚡ Loaded cached FAISS index,Index size:", embeddingIndex?.ntotal?.());
        } else {
            // Rebuild index from existing docs
            embeddingIndex = new IndexFlatL2({ dims: embeddingDim });
            for (const doc of docs) {
                const emb = await embedder(doc.content, { pooling: "mean", normalize: true });
                embeddings.push(Array.from(emb.data));
            }
            await embeddingIndex.add(embeddings);
            
            console.log(`⚡ Rebuilt FAISS index, ntotal: ${embeddingIndex?.ntotal?.()}`);
        }
    } else {
        // No cache: build from scratch
        try{
            const inputFolder = path.join(__dirname, "Input_JSON");
            docs = getAllDocsFromInputJSON(inputFolder);
            console.log(`⚡ Loaded ${docs.length} documents from Input_JSON`);
            embeddingIndex = new IndexFlatL2({ dims: embeddingDim });
            //await Promise.all(docs.map(embedDoc))

            for (const doc of docs) {
                // Skip anything that isn’t an object or missing content
                if (doc && typeof doc.content === "string" && doc.content.trim() !== "") {
                    const emb = await embedder(doc.content, { pooling: "mean", normalize: true });
                    embeddings.push(Array.from(emb.data));
                } else {
                    console.warn("⚠️ Skipping doc with invalid content:", doc);
                }
            }
            await embeddingIndex.add(embeddings);
            console.log(`⚡ Built new FAISS index, ntotal: ${embeddingIndex.ntotal()}`);
        }catch (err) {
            console.error("FAISS search failed:", err);
            return;
        }
    }

    // Save for next time
    fs.writeFileSync(metaPath,  JSON.stringify(docs), "utf-8");
    fs.writeFileSync(indexPath, JSON.stringify(embeddings), "utf-8");
    fs.writeFileSync(hashPath,  currentHash, "utf-8");
    console.log("FAISS index ready. Size:", embeddingIndex.ntotal());
    return embeddingIndex;
    } catch {
        throw new error("INITIALIZATION FAILED!!")
    }
}

// ---------------------------
// Retrieval
// ---------------------------
function retrieveTopK(queryVector, k = 5) {
    if (!embeddingIndex) return [];
    if (!queryVector || queryVector.length === 0) {
        console.warn("Empty query vector");
        return [];
    }

    if (embeddingIndex.ntotal() === 0) return [];
    try {
        const results = embeddingIndex.search(queryVector, k);
        if (!results?.labels) return [];
        return results.labels.map(i => docs[i]).filter(Boolean);
    } catch (err) {
        console.error("FAISS search failed:", err);
        return [];
    }
}

// ---------------------------
// Generation
// ---------------------------
/*async function generateLocal(prompt) {
    if(generator == null){
        await loadModels();
    }
    //CONFIG.tokenKey
    const result = await generator(prompt, { max_new_tokens: 200, temperature: 0.7 });

    if (!Array.isArray(result) || !result[0]?.generated_text) {
        console.warn("generateLocal returned invalid output", result);
        return "Error: LLM did not return text";
    }
    return result[0].generated_text;
}*/

const GRANITE_FALLBACK = "onnx-community/granite-4.0-350m-ONNX-web"; //"ibm-granite/granite-4.0-h-tiny" //path.resolve("granite-4.0-h-tiny-Q5_K_M.gguf");
//./resources/models/

async function generateLocal(prompt, modelPath = null) {
    const modelToUse = modelPath || CONFIG.model;

    let pipe = await pipeline("text-generation", GRANITE_FALLBACK, {
        //local_files_only: true,
        trust_remote_code: true,
        allowRemoteModels: true
    });
    try {
        pipe = await pipeline("text-generation", GRANITE_FALLBACK, {//path.join(LOCAL_MODEL_DIR, modelToUse), {
            //local_files_only: true,    // <- force offline
            trust_remote_code: true,
            allowRemoteModels: true
        });
    } catch (err) {
        console.warn(`Local model load failed for "${modelToUse}": ${err}. Using Granite fallback.`);
        pipe = await pipeline("text-generation", GRANITE_FALLBACK, {
            local_files_only: true,
            trust_remote_code: false,
            allowRemoteModels: true
        });
    }

    try {
        const result = await pipe(prompt, { max_new_tokens: 200, temperature: 0.7 });
        if (!Array.isArray(result) || !result[0]?.generated_text) {
            throw new Error("Pipeline returned invalid output");
        }
        return result[0].generated_text;
    } catch (err) {
        console.error(`Generation failed on "${modelToUse}": ${err}. Using Granite fallback.`);
        if (modelToUse !== GRANITE_FALLBACK) {
            const fallbackPipe = await pipeline("text-generation", GRANITE_FALLBACK, {
                local_files_only: true,
                trust_remote_code: false,
            });
            const fallbackResult = await fallbackPipe(prompt, { max_new_tokens: 200, temperature: 0.7 });
            return fallbackResult[0]?.generated_text || "Error: Granite fallback failed";
        }
        return "Error: Generation failed with Granite fallback";
    }
}

async function generateVerso(prompt) {
    generator = async function(prompt, options = {}) {
        return new Promise((resolve, reject) => {

            const proc = spawn("python", [
                "saguaro.py",
                "--prompt",
                prompt
            ], {
                env: {
                    ...process.env,
                    HIGHNOON_API_KEY: CONFIG.tokenKey
                }
            });

            let output = "";

            proc.stdout.on("data", d => output += d.toString());

            proc.on("close", code => {
                if (code !== 0) reject("Saguaro failed");
                else resolve([{ generated_text: output.trim() }]);
            });
        });
    };

    const result = await generator(prompt, { max_new_tokens: 200, temperature: 0.7 });
    if (!Array.isArray(result) || !result[0]?.generated_text) {
        console.error(`Grok generation failed: ${err}, falling back to Granite`);
        let resp = await generateGranite(prompt);
        return resp.output_text;
        
         // "Error: could not generate response";
        //console.warn("generateLocal returned invalid output", result);
        //return "Error: LLM did not return text";
    }
    return result[0].generated_text;
}

/*async function generateVerso(prompt) {
    if(generator == null){
        await loadModels();
    }
    //CONFIG.tokenKey
    const result = await generator(prompt, { max_new_tokens: 200, temperature: 0.7 });

    if (!Array.isArray(result) || !result[0]?.generated_text) {
        console.warn("generateLocal returned invalid output", result);
        return "Error: LLM did not return text";
    }
    return result[0].generated_text;
}*/


// Hardcoded path to the bundled model
const GRANITE_MODEL_PATH = "granite-4.0-h-tiny-Q5_K_M.gguf" //path.join(
    //process.resourcesPath,
    //"models",
    //"granite-4.0-h-tiny-Q5_K_M.gguf"
//);

async function generateGranite(prompt) {
    if (!fs.existsSync(GRANITE_MODEL_PATH)) {
        throw new Error("Granite model not found at " + GRANITE_MODEL_PATH);
    }

    // Initialize the pipeline with the hardcoded model
    const pipe = await pipeline("text-generation", GRANITE_MODEL_PATH);

    // Generate text
    const result = await pipe(prompt, { max_new_tokens: 200, temperature: 0.7 });

    if (!Array.isArray(result) || !result[0]?.generated_text) {
        console.warn("generateGranite returned invalid output", result);
        return "Error: LLM did not return text";
    }
    return result[0].generated_text;
}

async function generateGroq(prompt) {
    try {
        generator = new OpenAI({
            apiKey: CONFIG.tokenKey,
            baseURL:"https://api.groq.com/openai/v1",
        })

        let response = await generator.responses.create({
            input:prompt,
            model:CONFIG.model,
            temperature:CONFIG.temperature//,
            //tools:[],
            //citation_options: CONFIG.citation_options
        })

        console.log(response.output_text)
        console.log("Groq raw response:", response.output_text);
        return response.output_text;
    } catch (err) {
        console.error(`Groq generation failed: ${err}, falling back to Granite`);
        let resp = await generateGranite(prompt);
        return resp.output_text; // "Error: could not generate response";
    }
}

async function generateGrok(model, query) {
    try {
        generator = createXai({ apiKey: CONFIG.tokenKey });
        const { text } = await generateText({
            model:  generator.responses(model),
            system: 'You are Grok, a highly intelligent, helpful AI assistant.',
            prompt: query,
        });

        console.log(text)
        return text;
    } catch (err) {
        console.error(`Grok generation failed: ${err}, falling back to Granite`);
        let resp = await generateGranite(prompt);
        return resp.output_text; // "Error: could not generate response";
    }
}

async function generateResponse(model="grok-4-1-fast-reasoning", prompt) {
    switch (CONFIG.generationMode) {
        case "groq":
            return await generateGroq(prompt);
        case "grok":
            return await generateGrok(model, prompt);
        case "local":
            return await generateLocal(prompt);
        case "verso":
            return await generateVerso(prompt);
        default:
            return await generateGroq(prompt);
    }
}

// ---------------------------
// Public API
// ---------------------------
function detectIntent(prompt) {
    const p = prompt.toLowerCase();
    if (p.includes("image") || p.includes("picture") || p.includes("draw"))
        return "image";
    if (p.includes("3d") || p.includes("model") || p.includes("sdf"))
        return "3d";
    return "text";
}

function diversify(results, key = "Genre") {
    const seen = new Set();
    const output = [];
    for (const r of results) {
        if (!seen.has(r.metadata?.[key])) {
            seen.add(r.metadata?.[key]);
            output.push(r);
        }
    }
    return output;
}

function getModelsBySource(source) {
    return Object.fromEntries(
        Object.entries(modelRegistry.llm)
            .filter(([name, model]) => model.source === source)
    );
}


function setActiveStackLLM(llm) {
    activeStack.llm = llm;
}

async function sendMessage(userInput) {
    if (!embeddingIndex) {
        console.log("Initializing vector index...");
        embeddingIndex = await initialize();
    }

    try {
        console.log("STEP 1: received message");
        const intent = detectIntent(userInput);
        console.log("Intent:", intent);
        const generator = findGenerator(intent);

        if (generator) {
            console.log("Routing to generator:", generator.type, generator.model);

            switch (generator.type) {
                case "image":
                    return await runImageModel(generator.model, userInput, generator.source, generator.repo);
                case "3d":
                    return await run3DModel(generator.model, userInput, generator.source, generator.repo);
                case "voice":
                    return await runTTS(generator.model, userInput, generator.source, generator.repo);
                case "video":
                    return await runVideoModel(generator.model, userInput, generator.source, generator.repo);
            }
        }

        // TEXT PIPELINE (RAG + LLM)
        const llmModel = modelRegistry.llm[activeStack.llm];
        if (!llmModel) {
            throw new Error(`LLM "${activeStack.llm}" not found in registry`);
        }

        console.log("STEP 2: embedding user input");
        // STEP 2: enrich input with context variables
        const contextVars = {
            time: new Date().toLocaleTimeString(),
            day:  new Date().toLocaleDateString()
        };

        const fullUserInput   = `Time: ${contextVars.time}\nDay: ${contextVars.day}\nUser: ${userInput}`;
        let dc = new Decoder();
     
        // sanitize chat/day/year into a single string
        const filename = `chat_${contextVars.day}_${contextVars.time}.json`.replace(/[\\/:"*?<>|]/g, "_"); //_${contextVars.year}

        // write directly to Logs folder
        const filePath = path.join(__dirname, "Logs", filename);

        // make sure Logs exists (just the top folder)
        //await fs.mkdir(path.join(__dirname, "Logs"), { recursive: true });

        // write the log
        await fs.promises.writeFile(filePath, userInput, { encoding: 'utf-8' })
        const embeddingVector = await embedText(fullUserInput);

        console.log("STEP 3: retrieveTopK from FAISS");
        // STEP 3: retrieve docs from FAISS
        let retrievedDocs = retrieveTopK(embeddingVector, 10);

        console.log("STEP 4: applying diversity filter");
        // STEP 4: apply diversity filter
        retrievedDocs = diversify(retrievedDocs, "Genre"); // or another metadata key

        // STEP 5: build prompt for LLM
        const contextText = retrievedDocs.map(d => d.content).join("\n");
        const fullPrompt  = contextText
            ? `Context:\n${contextText}\n\nUser:\n${userInput}\n\nRespond naturally and helpfully.`
            : `User:\n${userInput}\n\nRespond naturally and helpfully.`;

        console.log("STEP 5: generating response");
        return await generateResponse(llmModel.repo, fullPrompt);

    } catch (err) {
        console.error("Error generating response:", err);
        return null;
    }
}

// Utility to reset conversation
function resetActiveStack() {
    activeConversationStack = [];
}

// Allow dynamic updates from frontend
function setConfig(newConfig) {
    CONFIG = { ...CONFIG, ...newConfig };
    console.log("CONFIG updated:", CONFIG);
}

/**
 * Generate a response for a specific agent, using its memory and RAG.
 *
 * @param {Object} agent - { id, model, systemPrompt }
 * @param {string} input - user input to the agent
 * @param {number} k - number of RAG documents to retrieve
 */
async function generateAgentResponse(agent, input, k = 5) {
    if (!agent || !agent.id) throw new Error("Agent must have an id");

    // ---------------------------
    // Embed the user input
    // ---------------------------
    const embeddingVector = await embedText(input);

    // ---------------------------
    // Retrieve context from agent-specific docs
    // ---------------------------
    let retrievedDocs = [];
    if (embeddingIndex && embeddingIndex.ntotal() > 0) {
        retrievedDocs = retrieveTopK(embeddingVector, k); // already filtered & sorted
    }

    const docContext = retrievedDocs
        .map(d => d?.content || d?.title || "")
        .filter(Boolean)
        .join("\n");

    // ---------------------------
    // Recall agent’s past memories
    // ---------------------------
    const memory = await RecallMemory(agent.id, input);
    const memoryContext = memory.results?.map(r => r.text).join("\n") || "";

    // ---------------------------
    // Build full prompt
    // ---------------------------
    const fullPrompt = `
        System Prompt:
        ${agent.systemPrompt || ""}

        Memory:
        ${memoryContext}

        Retrieved Context:
        ${docContext}

        User Input:
        ${input}

        Respond naturally and helpfully.
        `;

    // ---------------------------
    // Generate response
    // ---------------------------
    const response = await generateResponse(agent.model, fullPrompt);
    let resp = reflect(agent.id, response);

    // ---------------------------
    // Optionally store new memory
    // ---------------------------
    await RetainMemory(agent.id, response);
    return resp; //response;
}
// ---------------------------
// Exports
// ---------------------------

export  {
    setTokenKey,
    setTemperature,
    setContextWindowKey,
    setGenerationMode,
    saveDocument,
    loadDocument,
    deleteDocument,
    replicateDocument,
    mergeDocument,
    exportDocument,
    initialize,
    sendMessage,
    setConfig,
    updateCharacter,
    createGroqClient,
    setEngine, 
    getEngine,
    updateEngine,
    getEngineInstance,
    authorizeSheets,
    ingestSpreadsheetToFAISS,
    fetchSpreadsheet,
    ingestSheets,
    resetActiveStack,
    generateAgentResponse,
    getHFEmbedding,
    getOAIEmbedding,
    embeddings,
    pushToConversationStack,
    getConversationHistory,
    makeQRCode
};
