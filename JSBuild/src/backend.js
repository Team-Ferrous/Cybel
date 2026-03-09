// backend.js
const fs  = require("fs");
const path   = require("path");
const crypto = require("crypto");
const { app } = require('electron');
const {findGenerator} = require('./model_switcher')

const OpenAI = require("openai");
const { pipeline } = require("@xenova/transformers");
const { embeddingDim,
  embeddingIndex,
  Index, 
  IndexFlatL2, 
  IndexFlatIP 
} = require("./embeddings");
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
let embedder;
let generator;

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
const index = new IndexFlatL2(embeddingDim);

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
    try {
        console.log("Loading embedding model...");
        embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
        console.log("Embedding model loaded!");

        if (CONFIG.generationMode === "local") {
            console.log("Loading local text-generation model...");
            generator = await pipeline("text-generation", "Xenova/phi-2");
            console.log("Generator model loaded!");
        }
    } catch (err) {
        console.error("❌ Failed to load models:", err);
        alert("❌ Failed to load models:", err.message); // optional: show an Electron dialog or exit gracefully
        // dialog.showErrorBox("Model Load Error", err.message);
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
    await loadModels(); // load embedder & generator

    let embeddingIndex;
    const currentHash = computeDocumentsHash();
    const hasCache = fs.existsSync(indexPath) &&
                     fs.existsSync(metaPath) &&
                     fs.existsSync(hashPath);

    if (hasCache) {
        const savedHash = fs.readFileSync(hashPath, "utf-8");
        docs = JSON.parse(fs.readFileSync(metaPath, "utf-8"));

        if (savedHash === currentHash) {
            // Load cached embeddings
            const cachedEmbeddings = JSON.parse(fs.readFileSync(indexPath, "utf-8")); //IndexFlatL2.read(indexPath);
            embeddingIndex = new IndexFlatL2({ dims: embeddingDim });
            await embeddingIndex.add(cachedEmbeddings) //.add(cachedEmbeddings);
            console.log(`⚡ Loaded cached FAISS index, ntotal: ${embeddingIndex.ntotal()}`);
        } else {
            // Rebuild index from existing docs
            embeddingIndex = new Index({ type: 'HNSW', dims: embeddingDim });
            const embeddings = [];
            for (const doc of docs) {
                const emb = await embedder(doc.content, { pooling: "mean", normalize: true });
                embeddings.push(Array.from(emb.data));
            }
            await embeddingIndex.add(embeddings);
            console.log(`⚡ Rebuilt FAISS index, ntotal: ${embeddingIndex.ntotal()}`);
        }
    } else {
        // No cache: build from scratch
        try{
            const inputFolder = path.join(__dirname, "Input_JSON");
            docs = getAllDocsFromInputJSON(inputFolder);
            console.log(`⚡ Loaded ${docs.length} documents from Input_JSON`);
            embeddingIndex = new Index({ type: 'HNSW', dims: embeddingDim });
            const embeddings = [];
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
            return [];
        }
    }

    // Save for next time
    const embeddingsArray = await embeddingIndex.getAllVectors(); // pseudo-method
    fs.writeFileSync(metaPath, JSON.stringify(docs), "utf-8");
    fs.writeFileSync(indexPath, JSON.stringify(embeddingsArray), "utf-8");
    fs.writeFileSync(hashPath, currentHash, "utf-8");

    return embeddingIndex;
}
// ---------------------------
// Retrieval
// ---------------------------
async function retrieveTopK(queryVector, k = 5) {
    if (!embeddingIndex || embeddingIndex.ntotal === 0) return [];

    // queryVector must be a 1D array: [0.1, 0.2, ..., 0.384]
    try {
        const results = await embeddingIndex.search(queryVector, k);

        if (!results?.labels) return [];

        // map labels to docs
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
function detectIntent(prompt) {

    const p = prompt.toLowerCase();

    if (p.includes("image") || p.includes("picture") || p.includes("draw"))
        return "image";

    if (p.includes("3d") || p.includes("model") || p.includes("sdf"))
        return "3d";

    return "text";
}

async function sendMessage(userInput) {
    try {
        console.log("STEP 1: received message");
        const intent = detectIntent(userInput);
        console.log("Intent:", intent);
        const generator = findGenerator(intent);

        // GENERATOR PIPELINE
        if (generator) {

            console.log("Routing to generator:", generator.type);

            if (generator.type === "image")
                return await runImageModel(generator.model, userInput);

            if (generator.type === "3d")
                return await run3DModel(generator.model, userInput);

            if (generator.type === "voice")
                return await runTTS(generator.model, userInput);

            if (generator.type === "video")
                return await runVideoModel(generator.model, userInput);
        }

        // TEXT PIPELINE (RAG + LLM)
        console.log("STEP 2: retrieving context");
        const retrieved = await retrieveTopK(userInput, 10);
        console.log("STEP 3: retrieved docs:", retrieved.length);
        const context = retrieved
            .map(d => d?.content || d?.Entry || "")
            .filter(Boolean)
            .join("\n");

        console.log("STEP 4: building prompt");
        const fullPrompt = context
            ? `Context:\n${context}\n\nUser:\n${userInput}\n\nRespond naturally and helpfully.`
            : `User:\n${userInput}\n\nRespond naturally and helpfully.`;

        console.log("STEP 5: generating response");
        const response = await generateResponse(fullPrompt);
        console.log("STEP 6: done");
        return response;
    } catch (err) {
        console.error("Error generating response:", err);
        return null;
    }
}

/*async function sendMessage(userInput) {
    try {
        console.log("STEP 1: received message");

        if (userInput.toLowerCase().includes("image")) {
            return "IMAGE_DONE";
        }
        console.log("STEP 2: retrieving context");
        const retrieved = await retrieveTopK(userInput, 10);

        console.log("STEP 3: retrieved docs:", retrieved.length);

        const context = retrieved
            .map(d => d?.content || "")
            .filter(Boolean)
            .join("\n");

        console.log("STEP 4: building prompt");

        const fullPrompt = context
            ? `Context:\n${context}\n\nUser:\n${userInput}\n\nRespond naturally and helpfully.`
            : `User:\n${userInput}\n\nRespond naturally and helpfully.`;

        console.log("STEP 5: generating response");

        const response = await generateResponse(fullPrompt);

        console.log("STEP 6: done");

        return response;

    } catch (err) {
        console.error("Error generating response:", err);
        return null;
    }
}*/

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
    setEngine, 
    loadModel
};
