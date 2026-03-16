// backend.js
import fs         from "fs";
import crypto     from "crypto";
import electron   from "electron";
import Store from 'electron-store';
import path from 'path';

import { google } from "googleapis";
import { InstanceEngine } from "./instance_engine.js";

import OpenAI            from "openai";
import { createXai    }  from '@ai-sdk/xai';
import { generateText }  from 'ai';
import { error        }  from "node:console";

import { pipeline      } from "@xenova/transformers";
import { findGenerator, modelRegistry, activeStack } from './model_switcher.js'
import { embeddingDim, embeddingIndex, embedText } from "./embeddings.js";

import { dirname       } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { reflect, RecallMemory, RetainMemory } from "./hindsight.js";
import dotenv from "dotenv";
const { dialog, ipcMain } = electron;

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);
const require    = createRequire(import.meta.url);
const { IndexFlatL2 }  = require(path.resolve(__dirname, './node_modules/faiss-node/build/Release/faiss-node'));
dotenv.config({ path: path.join(__dirname, ".env") });

const TOKEN_PATH = path.join(__dirname, 'token.json');
const CREDENTIALS_PATH = path.join(__dirname, 'credentials.json');
const engine     = new InstanceEngine();
const SHEETS_TOKEN_PATH = path.join(__dirname, "sheets_token.json");
const SHEETS_CREDENTIALS_PATH = path.join(__dirname, "sheets_credentials.json"); // from GCP
let sheetsAuth;

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

        authWin.loadFile("oauth_code.html");

        ipcMain.once("google-sheets-code", (_, code) => {
            authWin.close();
            resolve(code);
        });

    });
}

async function authorizeSheets() {
    if (sheetsAuth) return sheetsAuth;

    const credentials = JSON.parse(fs.readFileSync(SHEETS_CREDENTIALS_PATH, "utf-8"));
    const { client_secret, client_id, redirect_uris } = credentials.installed;
    const oAuth2Client = new google.auth.OAuth2(client_id, client_secret, redirect_uris[0]);
    console.log("Authorizing Google Sheets access...");
    if (fs.existsSync(SHEETS_TOKEN_PATH)) {
        const token = JSON.parse(fs.readFileSync(SHEETS_TOKEN_PATH, "utf-8"));
        oAuth2Client.setCredentials(token);
    } else {
        // Here you’d need a simple input dialog in Electron to get the code
        const code = await requestOAuthCode();

        const { tokens } = await oAuth2Client.getToken(code);
        oAuth2Client.setCredentials(tokens);
        fs.writeFileSync(SHEETS_TOKEN_PATH, JSON.stringify(tokens), "utf-8");
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

// main.js
const store = new Store();

let CONFIG = {
    temperature:      0.75,
    contextWindow:    4096,
    generationMode:   "groq", //  "groq" | "local" | "grok" | "verso"
    model:            "openai/gpt-oss-20b",
    citation_options: 'enabled',
    groqKey:          store.get("groqKey") || null //localStorage.getItem("groqKey") //process.env.GROQ_API_KEY,
};

function createGroqClient() {
    return new Groq({ apiKey: CONFIG.groqKey });
}

//const DOCUMENT_DIR = path.join(app.getPath('userData'), 'documents');
const DOCUMENT_DIR = path.join(__dirname, "documents");

let embedder;
let generator;
let docs       = []
let embeddings = [];

// Active conversation stack
let activeConversationStack = []; // each entry: { user: string, bot: string }

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

function setTemperature(gen){
    CONFIG.temperature = gen;
}

function setContextWindowKey(key){
    CONFIG.contextWindow = key;
}

function setGenerationMode(gen){
    CONFIG.generationMode = gen;
}

function updateEngine(engine){
    CONFIG.model = engine;
}

function setGroqKey(key){
    CONFIG.groqKey = key;
    store.set("groqKey", key);
}

function setEngineInstance(eng){
    engine = eng;
}

function setEngine(conf){
    //eng = getEngineInstance();// || new InstanceEngine();
    CONFIG.engine = conf;
    store.set("current-engine", CONFIG.engine)
    // set engine to currently loaded pipeline + model set
    //setEngineInstance(eng);
}

function getEngine(conf){
    CONFIG.engine = conf;
    store.set("current-engine", CONFIG.engine)
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
        const emb = await embedder(text, { pooling: "mean", normalize: true });
        embeddings.push(Array.from(emb.data));
        docs.push({ content: text, metadata: row });
    }

    await embeddingIndex.add(embeddings);
    // Optional: persist for cache
    fs.writeFileSync(metaPath,  JSON.stringify(docs), "utf-8");
    fs.writeFileSync(indexPath, JSON.stringify(embeddings), "utf-8");
}

async function ensureDocumentDir() {
  await fs.mkdir(DOCUMENT_DIR, { recursive: true });
}

async function saveDocument(doc) {
  try {
    await ensureDocumentDir();
    const filePath = path.join(DOCUMENT_DIR, doc.title + ".json");
    await fs.promises.writeFile(filePath, JSON.stringify(doc, null, 2), "utf-8");
    return { success: true, path: filePath };
  } catch (err) {
    return { success: false, error: err.message };
  }
}

async function loadDocument(doc) {
  try {
        const result = await dialog.showOpenDialog({
            title: "Import Document",
            defaultPath: doc.title + ".txt",
            filters: [
            { name: "Text, JSON, PDF Files", extensions: ["txt", "json", "pdf"] }
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
                    const emb = await embedder(doc, { pooling: "mean", normalize: true });
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
        embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { token: hfToken }, model_file_name="all-MiniLM-L6-v2.pt");
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
    if (!embeddingIndex || embeddingIndex.ntotal() === 0) return [];
    try {
        const results = embeddingIndex.search([queryVector], k);
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
async function generateLocal(prompt) {
    if(generator == null){
        await loadModels();
    }
    const result = await generator(prompt, { max_new_tokens: 200, temperature: 0.7 });

    if (!Array.isArray(result) || !result[0]?.generated_text) {
        console.warn("generateLocal returned invalid output", result);
        return "Error: LLM did not return text";
    }
    return result[0].generated_text;
}


async function generateVerso(prompt) {
    if(generator == null){
        await loadModels();
    }
    const result = await generator(prompt, { max_new_tokens: 200, temperature: 0.7 });

    if (!Array.isArray(result) || !result[0]?.generated_text) {
        console.warn("generateLocal returned invalid output", result);
        return "Error: LLM did not return text";
    }
    return result[0].generated_text;
}

async function generateGroq(prompt) {
    try {
        let client = new OpenAI({
            apiKey: CONFIG.groqKey,
            baseURL:"https://api.groq.com/openai/v1",
        })

        let response = await client.responses.create({
            input:prompt,
            model:CONFIG.model,
            temperature:CONFIG.temperature//,
            //citation_options: CONFIG.citation_options
        })

        console.log(response.output_text)
        console.log("Groq raw response:", response.output_text);
        return response.output_text;
    } catch (err) {
        console.error("Groq generation failed:", err);
        return null;// "Error: could not generate response";
    }
}

async function generateGrok(model, query) {
    try {
        let client = createXai({ apiKey: process.env.XAI_API_KEY });
        const { text } = await generateText({
            model:  client.responses(model),
            system: 'You are Grok, a highly intelligent, helpful AI assistant.',
            prompt: query,
        });

        console.log(text)
        return text;
    } catch (err) {
        console.error("Groq generation failed:", err);
        return null;// "Error: could not generate response";
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
            day: new Date().toLocaleDateString()
        };

        const fullUserInput = `Time: ${contextVars.time}\nDay: ${contextVars.day}\nUser: ${userInput}`;
        const embeddingVector = await embedText(fullUserInput);

        // STEP 3: retrieve docs from FAISS
        let retrievedDocs = retrieveTopK(embeddingVector, 10);

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
    setGroqKey,
    setTemperature,
    setContextWindowKey,
    setGenerationMode,
    updateEngine,
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
    getEngineInstance,
    authorizeSheets,
    ingestSpreadsheetToFAISS,
    fetchSpreadsheet,
    ingestSheets,
    resetActiveStack,
    generateAgentResponse,
    embeddings
};
