import { embedText }  from "./embeddings.js"; // your embedding function
import { VestAuthClient } from "./vestauth.js";
const { IndexFlatL2 } = './node_modules/faiss-node/build/Release/faiss-node';

//example invocation
/*const engine = new InstanceEngine();
engine.spawn({
  id: "chat-agent",
  provider: "huggingface",
  secretKey: "HF_TOKEN",
  tools: ["search", "filesystem"]
});*/

//This is for the AI Agents Control Only
class InstanceEngine {
  constructor() {
    this.instances = new Map();
  }

  /**
   * Create a new instance with optional FAISS vector store
   */
  async spawn(config) {

    let vstAuth = new VestAuthClient();
    if (this.instances.has(config.id)) {
      return { success: false, error: "Instance already exists" };
    }

    const dimension = config.embeddingDim || 1536;
    const index = new IndexFlatL2(dimension);

    // Fetch provider secret if needed
    let providerToken = null;

    if (config.secretKey) {
      providerToken = await vstAuth.get(config.secretKey);
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

  get(id) {
    return this.instances.get(id);
  }

  async destroy(id) {
    const inst = this.instances.get(id);
    
    if (!inst) {
        return { success:false, error:"Instance not found" };
    }

    if (inst.shutdown) {
        await inst.shutdown();
    }

    this.inst.delete(id);

    return { success:true };
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

  async spawnAgent(config) {
    let vstAuth = new VestAuthClient();
    const agent = {
        id: config.id,
        model: config.model || "mistral7b",
        systemPrompt: config.systemPrompt || ""
    };

    // Persist settings
    //await vstAuth.set(`${agent.id}_model`, agent.model);
    //await vstAuth.set(`${agent.id}_systemPrompt`, agent.systemPrompt);
    return agent;
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


export {
 InstanceEngine
}