const { embedText, faiss } = require("./embeddings"); // your embedding function
const vestauth = require("./vestauth");

/*
npm i -g vestauth

vestauth agent init

Your agent sets secrets with a simple curl endpoint:
vestauth agent curl -X POST https://as2.dotenvx.com/set -d '{"KEY":"value"}'

And your agent gets secrets with a simple curl endpoint:
vestauth agent curl "https://as2.dotenvx.com/get?key=KEY"
*/

//This is for the AI Agents Control Only
class InstanceEngine {
  constructor() {
    this.instances = new Map();
  }

  /**
   * Create a new instance with optional FAISS vector store
   */

async spawn(config) {

  if (this.instances.has(config.id)) {
    return { success: false, error: "Instance already exists" };
  }

  const dimension = config.embeddingDim || 1536;
  const index = new faiss.IndexFlatL2(dimension);

  // Fetch provider secret if needed
  let providerToken = null;

  if (config.secretKey) {
    providerToken = await vestauth.get(config.secretKey);
  }

  const instance = {
    id: config.id,
    provider: config.provider,
    providerToken,
    tools: config.tools,
    mode: config.mode,
    sessionState: {},
    faissIndex: index,
    agent: null,
    createdAt: Date.now()
  };

  this.instances.set(config.id, instance);

  return { success: true, instance };
}

  /*async spawn(config) {
    if (this.instances.has(config.id)) {
      return { success: false, error: "Instance already exists" };
    }

    // Create a FAISS index per instance (L2 distance, 1536 dim for OpenAI embeddings)
    const dimension = config.embeddingDim || 1536;
    const index = new faiss.IndexFlatL2(dimension);

    const instance = {
      id: config.id,
      provider: config.provider,
      tools: config.tools,
      mode: config.mode,
      sessionState: {},
      faissIndex: index,
      agent: null, // future agent placeholder
      createdAt: Date.now()
    };

    this.instances.set(config.id, instance);

    return { success: true, instance };
  }*/

  get(id) {
    return this.instances.get(id);
  }

  delete(id) {
    const inst = this.instances.get(id);
    if (!inst) return;

    // Optionally free FAISS memory
    if (inst.faissIndex) inst.faissIndex.reset();

    this.instances.delete(id);
  }

  /**
   * Ingest a set of documents into an instance's FAISS index
   */
  async ingestDocuments(id, documents) {
    const inst = this.instances.get(id);
    if (!inst) throw new Error("Instance not found");

    for (const doc of documents) {
      // convert doc to embeddings
      const vector = await embedText(doc); // returns Float32Array
      inst.faissIndex.add([vector]);
    }

    return { success: true, count: documents.length };
  }

  /**
   * Attach an agent to an instance
   */
  attachAgent(id, agent) {
    const inst = this.instances.get(id);
    if (!inst) throw new Error("Instance not found");

    inst.agent = agent; // agent could be a function or object handling queries
  }

  /**
   * Query FAISS index for nearest neighbors
   */
  query(id, queryVector, k = 5) {
    const inst = this.instances.get(id);
    if (!inst) throw new Error("Instance not found");

    const result = inst.faissIndex.search([queryVector], k); // returns distances + indices
    return result;
  }
}

//example invocation
/*const engine = new InstanceEngine();
engine.spawn({
  id: "chat-agent",
  provider: "huggingface",
  secretKey: "HF_TOKEN",
  tools: ["search", "filesystem"]
});*/

module.exports = {
 InstanceEngine
}