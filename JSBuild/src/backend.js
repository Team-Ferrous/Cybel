// backend.js
import fs                from "fs";
import path              from "path";
import crypto            from "crypto";
import { dialog }        from 'electron';
import dotenv            from "dotenv";

import OpenAI            from "openai";
import { pipeline      } from "@xenova/transformers";
import { findGenerator } from './model_switcher.js'
import { embeddingDim, embeddingIndex, embedText } from "./embeddings.js";

import { dirname }       from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);
const require    = createRequire(import.meta.url);
const { IndexFlatL2 }  = require(path.resolve(__dirname, './node_modules/faiss-node/build/Release/faiss-node'));

dotenv.config({ path: path.join(__dirname, ".env") });

// main.js
import Store from 'electron-store';
import { error } from "node:console";
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
let embeddingIndex;
let docs       = []
let embeddings = [];

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

function setEngine(conf){
    CONFIG.engine = conf;
    store.set("current-engine", CONFIG.engine)
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
            { name: "Text Files", extensions: ["txt"] }
            ]
        });

        if (result.canceled) {
            return { success: false, canceled: true };
        }

        const importPath = result.filePaths[0];
        const data       = await fs.readFile(importPath , "utf-8");
        return { success: true, path: importPath, document: JSON.parse(data) };
    } catch (err) {
        return { success: false, error: err.message };
    }
}

async function deleteDocument(doc) {
  try {
    const result = await dialog.showMessageBox({
            title: "Select Document to Delete",
            defaultPath: doc.title + ".txt",
            filters: [
            { name: "Text Files", extensions: ["txt"] }
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
            defaultPath: doc.title + ".txt",
            filters: [
            { name: "Text Files", extensions: ["txt"] }
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
        console.log("Loading embedding model...");
        embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { token: hfToken });
        console.log("Embedding model loaded!");

        if (CONFIG.generationMode === "groq") {
            console.log("Loading Groq (requested) model...");
            generator = await pipeline("text-generation", "Xenova/phi-2", { token: hfToken });
            console.log("Generator model loaded!");
        }
        if (CONFIG.generationMode === "grok") {
            console.log("Loading Grok model...");
            generator = await pipeline("text-generation", "Xenova/phi-2", { token: hfToken });
            console.log("Generator model loaded!");
        }
        if (CONFIG.generationMode === "local") {
            console.log("Loading local text-generation model...");
            generator = await pipeline("text-generation", "Xenova/phi-2", { token: hfToken });
            console.log("Generator model loaded!");
        }
        if (CONFIG.generationMode === "verso") {
            console.log("Loading Verso model...");
            generator = await pipeline("text-generation", "Xenova/phi-2", { token: hfToken });
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

async function sendMessage(userInput, sessionId="default") {
    if (!embeddingIndex) embeddingIndex = await initialize();

    const convoPath = path.join(CONVO_DIR, sessionId + ".json");
    let history = [];
    if (fs.existsSync(convoPath)) {
        history = JSON.parse(fs.readFileSync(convoPath, "utf-8"));
    }

    // Append user message
    history.push({ role: "user", content: userInput, timestamp: Date.now() });

    // Retrieve context
    const embeddingVector = await embedText(userInput);
    let retrievedDocs = [];
    if (embeddingIndex && embeddingIndex.ntotal() > 0) {
        const results = embeddingIndex.search([embeddingVector], 10);
        retrievedDocs = results.labels.map(i => docs[i]).filter(Boolean);
    }

    const context = retrievedDocs.map(d => d?.content || "").join("\n");
    const convoContext = history.map(m => `${m.role}:\n${m.content}`).join("\n");

    const fullPrompt = `${context ? "Context:\n" + context + "\n\n" : ""}${convoContext}\n\nRespond naturally and helpfully.`;

    const response = await generateResponse(fullPrompt);

    // Append assistant response
    history.push({ role: "assistant", content: response, timestamp: Date.now() });
    fs.writeFileSync(convoPath, JSON.stringify(history, null, 2), "utf-8");

    return response;
}


// Allow dynamic updates from frontend
function setConfig(newConfig) {
    CONFIG = { ...CONFIG, ...newConfig };
    console.log("CONFIG updated:", CONFIG);
}

function loadModel(modelName){

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
    loadModel
};
