import { embedText }      from "./embeddings.js"; // your embedding function
import { VestAuthClient } from "./vestauth.js";
const  { IndexFlatL2 } = './node_modules/faiss-node/build/Release/faiss-node';
import { dialog }   from "electron";

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
    defaultAgentConfig = {
        id: 0,
        model:        "mistral7b",
        systemPrompt: "You are a helpful assistant. Use the tools at your disposal to answer the user's query."
    }

  constructor() {
    this.config    = this.defaultAgentConfig;
    this.instances = new Map();
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
      secretKey: providerToken,
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

  async spawnAgent(config = null) {
    let vstAuth = new VestAuthClient();
    const agentConfig = config || this.defaultAgentConfig;
    if(this.instances.size >= 0) { 
      const agent = {
          id:           agentConfig.id, // simple incremental ID
          model:        agentConfig.model        || "mistral7b",
          systemPrompt: agentConfig.systemPrompt || ""
      };

        // Persist settings
        //await vstAuth.set(`${agent.id}_model`, agent.model);
        //await vstAuth.set(`${agent.id}_systemPrompt`, agent.systemPrompt);
        return agent;
    }
    else{
      const agent = {
          id:           this.instances.size + 1, // simple incremental ID
          model:        agentConfig.model        || "mistral7b",
          systemPrompt: agentConfig.systemPrompt || ""
      };

      // Persist settings
      //await vstAuth.set(`${agent.id}_model`, agent.model);
      //await vstAuth.set(`${agent.id}_systemPrompt`, agent.systemPrompt);
      return agent;
    }
  }

  /**
   * Attach an agent to an instance
   */
  attachAgent(id, agent) {
    const inst = this.instances.get(id);
    if (!inst) throw new Error("Instance not found");

    inst.agent = agent; // agent could be a function or object handling queries
    return { success:true };
  }

  detachAgent(id) {
    const inst = this.instances.get(id);
    inst.agent = null; // Remove the attached agent
    return { success:true };
  }
  
  async saveConfig(doc) {
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

  async loadConfig() {
  try {
        const result = await dialog.showOpenDialog({
            title: "Import Document",
            defaultPath: "default.json",
            filters: [
              {name: "JSON", extensions: [ "json" ]},
            ]
        });

        if (result.canceled) {
            return { success: false, canceled: true };
        }
        const importPath = result.filePaths[0];
        const data       = await fs.readFile(importPath , "utf-8");
        if (importPath.endsWith(".json")) {
          this.config = JSON.parse(data);
          return { success: true, path: importPath, config: this.config };
        }
        else {
          return { success: false, error: "Unsupported file type" };
        }
    } catch (err) {
        return { success: false, error: err.message };
    }
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