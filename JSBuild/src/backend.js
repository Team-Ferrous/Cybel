import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";
import { pipeline } from "@xenova/transformers";
import { HierarchicalNSW } from "hnswlib-node";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------
// Paths
// ---------------------------

const inputDir = path.join(__dirname, "Input_JSON");
const logsDir  = path.join(__dirname, "Logs");

const indexPath = path.join(__dirname, "vector.index");
const metaPath  = path.join(__dirname, "vector_docs.json");
const hashPath  = path.join(__dirname, "doc_hash.txt");

if (!fs.existsSync(inputDir)) fs.mkdirSync(inputDir, { recursive: true });
if (!fs.existsSync(logsDir))  fs.mkdirSync(logsDir,  { recursive: true });

// ---------------------------
// Globals
// ---------------------------

let embedder = null;
let embeddingIndex = null;
let embeddingDim = 0;
let docs = [];

// ---------------------------
// Hash Helper
// ---------------------------

function computeDocumentsHash() {
    const hasher = crypto.createHash("sha256");

    const files = fs.readdirSync(inputDir)
        .filter(f => f.endsWith(".json"))
        .sort();

    for (const file of files) {
        const buffer = fs.readFileSync(path.join(inputDir, file));
        hasher.update(buffer);
    }

    return hasher.digest("hex");
}

// ---------------------------
// Initialize
// ---------------------------

export async function initialize() {

    console.log("Loading embedding model...");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

    const currentHash = computeDocumentsHash();

    if (
        fs.existsSync(indexPath) &&
        fs.existsSync(metaPath) &&
        fs.existsSync(hashPath)
    ) {
        const savedHash = fs.readFileSync(hashPath, "utf-8");

        if (savedHash === currentHash) {
            console.log("⚡ Loading cached index...");

            docs = JSON.parse(fs.readFileSync(metaPath, "utf-8"));

            embeddingDim = 384;
            embeddingIndex = new HierarchicalNSW("cosine", embeddingDim);
            embeddingIndex.readIndex(indexPath);

            console.log("✅ Index ready.");
            return;
        }
    }

    console.log("🛠 Building index...");

    docs = [];

    const files = fs.readdirSync(inputDir).filter(f => f.endsWith(".json"));

    for (const file of files) {
        const data = JSON.parse(
            fs.readFileSync(path.join(inputDir, file), "utf-8")
        );

        for (const item of data) {
            docs.push({
                content: JSON.stringify(item),
                source: file
            });
        }
    }

    if (docs.length === 0) {
        console.log("⚠️ No documents found.");
        return;
    }

    const embeddings = [];

    for (const doc of docs) {
        const output = await embedder(doc.content, {
            pooling: "mean",
            normalize: true
        });
        embeddings.push(output.data);
    }

    embeddingDim = embeddings[0].length;

    embeddingIndex = new HierarchicalNSW("cosine", embeddingDim);
    embeddingIndex.initIndex(docs.length);

    embeddings.forEach((vec, i) => {
        embeddingIndex.addPoint(vec, i);
    });

    embeddingIndex.writeIndex(indexPath);
    fs.writeFileSync(metaPath, JSON.stringify(docs));
    fs.writeFileSync(hashPath, currentHash);

    console.log("✅ Index built and saved.");
}

// ---------------------------
// Retrieval
// ---------------------------

async function retrieveTopK(query, k = 5) {
    if (!embeddingIndex) return [];

    const output = await embedder(query, {
        pooling: "mean",
        normalize: true
    });

    const result = embeddingIndex.searchKnn(output.data, k);
    return result.neighbors.map(i => docs[i]);
}

// ---------------------------
// Public API
// ---------------------------

export async function sendMessage(userInput) {

    if (!embeddingIndex) {
        throw new Error("Backend not initialized.");
    }

    if (userInput.toLowerCase().includes("image")) {
        return "IMAGE_DONE";
    }

    const retrieved = await retrieveTopK(userInput, 10);
    const context = retrieved.map(d => d.content).join("\n");

    // Replace this with your LLM generation logic
    const response = `Context:\n${context}\n\nUser: ${userInput}`;

    return response;
}
