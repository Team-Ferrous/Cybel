const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
    sendMessage:          (msg)  => ipcRenderer.invoke("chat:send",                 msg),
    setMode:              (mode) => ipcRenderer.invoke("chat:setMode",              mode),
    setGroqKey:           (key)  => ipcRenderer.invoke("chat:setGroqKey",           key),
    setTemperature:       (key)  => ipcRenderer.invoke("chat:setTemperature",       key),
    setContextWindowKey:  (key)  => ipcRenderer.invoke("chat:setContextWindowKey",  key),
    updateEngine:         (key)  => ipcRenderer.invoke("engine:update",             key),
    saveDocument:         (doc)  => ipcRenderer.invoke("engine:save_document",      doc),
    loadDocument:         (doc)  => ipcRenderer.invoke("engine:load_document",      doc),
    deleteDocument:       (doc)  => ipcRenderer.invoke("engine:delete_document",    doc),
    replicateDocument:    (doc)  => ipcRenderer.invoke("engine:replicate_document", doc),
    mergeDocument:        (doc)  => ipcRenderer.invoke("engine:merge_document",     doc),
    exportDocument:       (doc)  => ipcRenderer.invoke("engine:export_document",    doc),
    updateCharacter:      (character) => ipcRenderer.invoke("engine:updateCharacter", character),
    spawnInstance: (config) => ipcRenderer.invoke("engine:spawn_instance", config),
    openFileDialog: () => ipcRenderer.invoke('open-file-dialog'),
    saveFileDialog: () => ipcRenderer.invoke('save-file-dialog'),
    DecodeRawDirectory: (dir) =>
      ipcRenderer.invoke('decode-directory', {
        dir,
        options: { mode: "raw" }
    }),
    DecodeChatDirectory: (dir) =>
      ipcRenderer.invoke('decode-directory', {
        dir,
        options: { mode: "chat", includeMetadata: true }
    }),
    ingestDocuments: (paths) => ipcRenderer.invoke("rag:ingest", paths),
    engineIngestChats: (instanceId, files) => ipcRenderer.invoke("engine:ingest_documents", { instanceId, files }),
    loadModel: (modelName) => ipcRenderer.invoke("engine:loadModel", { modelName }),
    // AGENT WORKFLOW
    spawnAgent: (config) => ipcRenderer.invoke("engine:spawnAgent", config),
    getAgent: (id)       => ipcRenderer.invoke("engine:getAgent",       id),
    attachAgent: (id, agent)         => ipcRenderer.invoke("engine:attachAgent", id, agent),
    queryAgent:  (id, vector, k)      => ipcRenderer.invoke("query",              id, vector, k),
    ingestAgentDocuments: (id, docs) => ipcRenderer.invoke("ingestDocuments", id, docs),
    hfLogin: () => ipcRenderer.invoke('hf-login'),
    msLogin: () => ipcRenderer.invoke('ms-login')
  });
