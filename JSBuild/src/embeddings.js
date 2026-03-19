// embeddings.js
import path              from 'path';
import { dirname }       from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

// get __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);
const require    = createRequire(import.meta.url);
const { IndexFlatL2, IndexHNSWFlat, Index, IndexFlatIP } = require(path.resolve(__dirname, './node_modules/faiss-node/build/Release/faiss-node'));

// docs array must include embedding with each doc
// Example: docs = [{ content: 'text', embedding: [0.1, ...] }, ...]
// Initialize HNSW or Flat index (dims = embedding length)
const embeddingDim = 384; // match your model embeddings
let embeddingIndex = new IndexFlatL2(embeddingDim); //new IndexHNSWFlat(embeddingDim, 32); // new IndexFlatL2({ type: 'HNSW', dims: embeddingDim }); //IndexFlatIP
let docs = [
  { content: "Hello world",     embedding: [0.1, 0.2, 0.3 /* ... match embeddingDim */] },
  { content: "Another example", embedding: [0.4, 0.5, 0.6 /* ... match embeddingDim */] },
  // add more documents as needed
];
let embeddings = docs.map(d => d.embedding);

// Add embeddings to the index
console.log("FAISS index loaded, ntotal:", embeddingIndex.ntotal);

function initFAISS() {
  embeddingIndex = new faiss.IndexFlatL2(embeddingDim);
  docs = [];
  console.log("⚡ FAISS index initialized with dimension", embeddingDim);
}

/**
 * Add vectors and metadata to FAISS
 * @param {Float32Array[]} vectors 
 * @param {any[]} metadatas 
 */
function addToFAISS(vectors, metadatas) {
  if (!embeddingIndex) throw new Error("FAISS index not initialized");
  if (vectors.length !== metadatas.length)
    throw new Error("Vectors and metadata count mismatch");

  const flatEmbeddings = new Float32Array(vectors.length * embeddingDim);
  vectors.forEach((vec, i) => flatEmbeddings.set(vec, i * embeddingDim));

  embeddingIndex.add(flatEmbeddings, vectors.length);
  docs.push(...metadatas);

  console.log(`✅ Added ${vectors.length} vectors to FAISS index`);
}

/**
 * Search top-k nearest neighbors
 * @param {Float32Array} query 
 * @param {number} k 
 */
function searchFAISS(query, k = 5) {
  if (!embeddingIndex) throw new Error("FAISS index not initialized");

  const distances = new Float32Array(k);
  const labels = new BigInt64Array(k);

  embeddingIndex.search(query, k, distances, labels);

  return Array.from(labels)
    .map(i => docs[Number(i)])
    .filter(Boolean);
}

function embedText(doc, size = embeddingDim) {
    const vec = new Float32Array(size);

    const text = doc.toLowerCase();

    for (let i = 0; i < text.length; i++) {
        const code = text.charCodeAt(i);
        const index = code % size;
        vec[index] += 1;
    }

    // normalize vector (L2)
    let norm = 0;
    for (let i = 0; i < size; i++) {
        norm += vec[i] * vec[i];
    }

    norm = Math.sqrt(norm);

    if (norm > 0) {
        for (let i = 0; i < size; i++) {
            vec[i] /= norm;
        }
    }

  // Optional: normalize for cosine similarity
  const normC = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
  if (norm > 0) {
    for (let i = 0; i < size; i++) vec[i] /= normC;
  }

  return vec;
}

export {
  initFAISS,
  addToFAISS,
  searchFAISS,
  embedText,
  embeddingDim,
  IndexFlatL2,
  Index, 
  IndexFlatIP,
  embeddingIndex,
  embeddings 
};