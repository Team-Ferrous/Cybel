// backend.js
const fs  = require("fs");
const path   = require("path");
const crypto = require("crypto");
const { app } = require('electron');

const OpenAI = require("openai");
const { pipeline } = require("@xenova/transformers");
const { faiss } = require("./embeddings");
require("dotenv").config({ path: path.join(__dirname, ".env") });

// main.js
const Store = require('electron-store').default;
const store = new Store();

let CONFIG = {
    temperature:      0.75,
    contextWindow:    4096,
    generationMode:   "groq", //  "groq" | "local"
    model:            "openai/gpt-oss-20b",
    citation_options: 'enabled',
    groqKey:          store.get("groqKey") || null //localStorage.getItem("groqKey") //process.env.GROQ_API_KEY,
};

function createGroqClient() {
    return new Groq({ apiKey: CONFIG.groqKey });
}

//const DOCUMENT_DIR = path.join(__dirname, "documents");
const DOCUMENT_DIR = path.join(app.getPath('userData'), 'documents');


let embedder = null;
let generator = null;
let embeddingIndex = null;
let embeddingDim = 384;
let docs = [];

// ---------------------------
// Paths
// ---------------------------
const inputDir  = path.join(__dirname, "Input_JSON");
const logsDir   = path.join(__dirname, "Logs");
const indexPath = path.join(__dirname, "vector.index");
const metaPath  = path.join(__dirname, "vector_docs.json");
const hashPath  = path.join(__dirname, "doc_hash.txt");

if (!fs.existsSync(inputDir)) fs.mkdirSync(inputDir, { recursive: true });
if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });
const index = new faiss.IndexFlatL2(embeddingDim);



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
        const result = await dialog.showSaveDialog({
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
        const filePath   = path.join(DOCUMENT_DIR, doc.title + ".json");
        const data       = await fs.readFile(filePath, "utf-8");
        return { success: true, path: importPath, document: JSON.parse(data) };
    } catch (err) {
        return { success: false, error: err.message };
    }
}

async function deleteDocument(doc) {
  try {
    const result = await dialog.showSaveDialog({
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
        const filePath   = path.join(DOCUMENT_DIR, doc.title + ".json");
        await fs.unlink(filePath);
        return { success: true, path: importPath };
    } catch (err) {
        return { success: false, error: err.message };
    }
}

async function replicateDocument(doc) {
  try {
    const sourcePath = path.join(DOCUMENT_DIR, doc.title + ".json");
    const targetPath = path.join(DOCUMENT_DIR, doc.newTitle + ".json");

    const result = await dialog.showSaveDialog({
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
    console.log("Loading embedding model...");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

    if (CONFIG.generationMode === "local") {
        console.log("Loading local text-generation model...");
        generator = await pipeline("text-generation", "Xenova/phi-2");
    }
}

// ---------------------------
// Initialize / Build Index
// ---------------------------
async function initialize() {
    try {
        await loadModels();
        const currentHash = computeDocumentsHash();

        // Load cached index if possible
        if (fs.existsSync(indexPath) && fs.existsSync(metaPath) && fs.existsSync(hashPath)) {
            const savedHash = fs.readFileSync(hashPath, "utf-8");
            if (savedHash === currentHash) {
                docs = JSON.parse(fs.readFileSync(metaPath, "utf-8"));

                // load cached embeddings if you saved them somewhere
                // const flatEmbeddings = ...;

                embeddingIndex = new faiss.IndexFlatL2(embeddingDim);
                // embeddingIndex.add(flatEmbeddings); <-- only if you have embeddings cached
                console.log("⚡ Loaded cached index.");
                return;
            }
        }

        // Rebuild index
        console.log("🛠 Building new vector index...");
        docs = [];
        const files = fs.readdirSync(inputDir).filter(f => f.endsWith(".json"));

        for (const file of files) {
            const raw = fs.readFileSync(path.join(inputDir, file), "utf-8");
            const data = JSON.parse(raw);

            if (Array.isArray(data)) {
                for (const item of data) {
                    docs.push({ content: JSON.stringify(item), source: file });
                }
            } else if (typeof data === "object" && data !== null) {
                docs.push({ content: JSON.stringify(data), source: file });
            } else {
                console.warn(`⚠️ File ${file} contains unsupported JSON type, skipping`);
            }
        }

        if (docs.length === 0) {
            console.warn("⚠️ No documents found. Skipping embedding index.");
            return;
        }

        // Compute embeddings
        const embeddings = [];
        for (const doc of docs) {
            const output = await embedder(doc.content, { pooling: "mean", normalize: true });
            embeddings.push(output.data); // Float32Array
        }

        // Flatten embeddings for FAISS
        const numVectors = embeddings.length;
        const flatEmbeddings = new Float32Array(numVectors * embeddingDim);
        embeddings.forEach((vec, i) => {
            if (vec.length !== embeddingDim) throw new Error(`Embedding vector length mismatch`);
            flatEmbeddings.set(vec, i * embeddingDim);
        });

        // ✅ Only one FAISS index and one add
        // embeddings: Array of Float32Array
        // Add embeddings
        const arrayEmbeddings = embeddings.map(vec => Array.from(vec)); // number[][]
        arrayEmbeddings.forEach(vec => index.add(vec)); // add one by one


        // If your binding supports batch add, you could just do:
        // index.add(arrayEmbeddings);
        // Save the index for later
        index.write(indexPath);
        fs.writeFileSync(metaPath, JSON.stringify(docs));
        fs.writeFileSync(hashPath, currentHash);
        console.log(`✅ FAISS index built with ${docs.length} documents.`);
    } catch (err) {
        console.error("Initialization failed:", err);
    }
}

// ---------------------------
// Retrieval
// ---------------------------
async function retrieveTopK(query, k = 5) {
    if (!embeddingIndex || !embedder) return [];

    const output = await embedder(query, { pooling: "mean", normalize: true });

    if (!output?.data) return [];
    const result = embeddingIndex.searchKnn(output.data, k);

    if (!result?.neighbors) return [];
    return result.neighbors.map(i => docs[i]).filter(Boolean);
}

// ---------------------------
// Generation
// ---------------------------
async function generateLocal(prompt) {
    if(generator == null){
        loadModels();
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

async function generateResponse(prompt) {
    switch (CONFIG.generationMode) {
        case "groq":
            return await generateGroq(prompt);
        case "local":
            return await generateLocal(prompt);
        default:
            return await generateGroq(prompt);
    }
}

// ---------------------------
// Public API
// ---------------------------
async function sendMessage(userInput) {
    try {
        if (userInput.toLowerCase().includes("image")) return "IMAGE_DONE";

        const retrieved = await retrieveTopK(userInput, 10);
        const context   = retrieved.map(d => d.content).join("\n");

        const fullPrompt = `
            Context:
            ${context}

            User:
            ${userInput}

            Respond naturally and helpfully.
        `;

        return await generateResponse(fullPrompt);
    } catch (err) {
        console.error("Error generating response:", err);
        return null;
    }
}

// Allow dynamic updates from frontend
function setConfig(newConfig) {
    CONFIG = { ...CONFIG, ...newConfig };
    console.log("CONFIG updated:", CONFIG);
}

// ---------------------------
// Exports
// ---------------------------
module.exports = {
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
    setEngine
};
