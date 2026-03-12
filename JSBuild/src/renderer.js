// renderer.js
import { hub } from './model_switcher.js';

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


document.querySelectorAll("input[type='range']").forEach(slider => {
slider.addEventListener("input", (e) => {
    const value = e.target.value;

    if (e.target.name === "temperature") {
        window.api.setTemperature({ temperature: parseFloat(value) });
    }

    if (e.target.name === "contextWindowKey") {
        window.api.setContextWindowKey({ temperature: parseFloat(value) });
    }

    if (e.target.name === "context") {
        window.api.updateEngine({ contextWindow: parseInt(value) });
    }
});
});

function loadAppearance() {
const stored = localStorage.getItem("appearanceConfig");
if (!stored) return;

const { accent, theme, background } = JSON.parse(stored);

    // Apply accent
    if (accent) document.documentElement.style.setProperty('--accent', accent);

    // Apply theme
    if (theme) document.body.dataset.theme = theme;
    const avatarImage = document.getElementById("avatarImage");
    const colorSwatches = document.querySelectorAll('[data-color]');
    colorSwatches.forEach(swatch => {
    swatch.addEventListener("click", () => {
            const color = swatch.dataset.color;
            document.documentElement.style.setProperty("--accent", color);
            const bgSelectS = document.getElementById('bgSelector');
            switchBG(bgSelectS.value)

            // update selected ring
            colorSwatches.forEach(s => 
            s.classList.remove("ring-4", "ring-white")
            );
            swatch.classList.add("ring-4", "ring-white");
        });
    });
    // Apply background
    const bgSelect = document.getElementById('bgSelector'); // <-- define it
    if (background && bgSelect) {
        bgSelect.value = background;
        switchBG(background); // your existing function
        saveAppearance();
    }

    // Make inline body styles match
    document.body.style.backgroundColor = getComputedStyle(document.body)
                                            .getPropertyValue('--bg-primary');
    document.body.style.color = getComputedStyle(document.body)
                                    .getPropertyValue('--text-primary');
}

// Run after DOM is ready
document.addEventListener("DOMContentLoaded", loadAppearance);
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-action]").forEach(btn => {
    btn.addEventListener("click", async () => {
        const action = btn.dataset.action;
        const files = await ipcRenderer.invoke("open-file-dialog");
        if (!files || files.length === 0) return;
        console.log(action, files);
        });
    });

    document.querySelector("#saveBtn").addEventListener("click", async () => {
        const files = await window.api.saveFileDialog();
        console.log(files);
    });

    document.querySelector("#loadBtn").addEventListener("click", async () => {
            const files = await window.api.openFileDialog();
            if (!files || files.length === 0) return;

            const instanceId = "default"; // or let the user select/create an instance
            const result = await window.api.ingestDocuments(instanceId, files);
            if (result.success) {
                console.log(`Ingested ${result.ingested} chunks into instance ${instanceId}`);
            } else {
                console.error("Ingestion failed:", result.message);
            }
    });
})

    // -------------------------------
    // Local Model Selector Manager
    // -------------------------------
    const localSelector = document.getElementById("local-model-selector");

    // Keep a list of installed local models
    let installedModels = [];
    function loadModel(modelName) {

        // Running inside Electron
        if (window.api && typeof window.api.loadModel === "function") {
            window.api.loadModel(modelName);
            return;
        }

        // Fallback for browser / dev mode
        console.log("Model load requested:", modelName);
    }

    // Initialize dropdown: mark installed models green, others red
    async function initLocalSelector() {
        installedModels =  []
        
        // TBA: make Installed Models call this and apply the results 
        //await modelHub.getLocalModels(); // ["Mistral 7b", "dolphin3", ...]
        Array.from(localSelector.options).forEach(option => {
            const isLocal      = installedModels.includes(option.text);
            option.style.color = isLocal ? "limegreen" : "red";
            option.title       = isLocal ? "Loaded locally" : "Available remotely";

            // auto-load green models
            if (isLocal) {
                loadModel(option.text);
            }
        });
    }

    // Update option color / status dynamically
    function updateOptionStatus(modelName, isLocal) {
        const option = Array.from(localSelector.options).find(o => o.text === modelName);
        if (!option) return;

        option.style.color = isLocal ? "limegreen" : "red";
        option.title = isLocal ? "Loaded locally" : "Available remotely";

        // Load the model immediately if it became local
        if (isLocal) {
            loadModel(modelName);
        }
    }

    // Add remote models dynamically (from search results)
    function addRemoteModel(modelName) {
        const exists = Array.from(localSelector.options).some(o => o.text === modelName);
        if (exists) return;

        const opt = document.createElement("option");
        opt.text = modelName;
        opt.style.color = "red";
        opt.title = "Available remotely";
        localSelector.appendChild(opt);
    }

    // Handle a model being downloaded
    async function onModelDownloaded(modelName) {
        // Update local list
        if (!installedModels.includes(modelName)) installedModels.push(modelName);

        // Update UI
        updateOptionStatus(modelName, true);

        // Register in GeneratorRegistry
        registerGenerator(modelName);

        // Load into memory
        await loadModel(modelName);
    }

    // Event listener for user selecting a model from dropdown
    localSelector.addEventListener("change", async (e) => {
        const selected = e.target.value;

        if (!installedModels.includes(selected)) {
            console.log(`Model "${selected}" is not downloaded yet!`);
            return;
        }

        console.log(`Switching to model: ${selected}`);
        await loadModel(selected); // ensure runtime has it loaded
    });

    // Call this on app startup
    initLocalSelector();

    const input  = document.getElementById("groq-key-input");
    const button = document.getElementById("groq-key-confirm");
    const modeSelector = document.getElementById("modeSelector");

    const groqPanels      = document.getElementById("groq-panels");
    const localOptsPanels = document.getElementById("local-option-panel");

    const localPanels = document.getElementById("local-panels");
    const grokPanels = document.getElementById("grok-panels");
    const versoPanels = document.getElementById("verso-panels");
    const versoOptsPanels = document.getElementById("verso-option-panel");

    const hf = document.getElementById("source-hf");
    const ms = document.getElementById("source-ms");
    hf.addEventListener("change", () => {
        if (hf.checked) ms.checked = false;
    });
    ms.addEventListener("change", () => {
        if (ms.checked) hf.checked = false;
    });

    const vsModeSelector = document.getElementById("vs-model-selector");
    vsModeSelector.addEventListener("change", (e) => {
        const vs = e.target.value;
        if (versoOptsPanels && vs.includes("High Noon")) {
            versoOptsPanels.style.display = "block";
        } else {
            versoOptsPanels.style.display = "none";
        }
    });


    modeSelector.addEventListener("change", (e) => {
        const mode = e.target.value;
        if (mode === "groq") {
            if (groqPanels)  groqPanels.style.display        = "block";
            if (localOptsPanels) localOptsPanels.style.display   = "none";
            if (localPanels) localPanels.style.display  = "none";
            if (grokPanels)  grokPanels.style.display   = "none";
            if (versoPanels) versoPanels.style.display  = "none";
            if(versoOptsPanels) versoOptsPanels.style.display = "none"
        }
        if(mode === "grok"){
            if (groqPanels)  groqPanels.style.display  = "none";
            if (localOptsPanels) localOptsPanels.style.display   = "none";
            if (localPanels) localPanels.style.display = "none";
            if (grokPanels)  grokPanels.style.display  = "block";
            if (versoPanels) versoPanels.style.display  = "none";
            if(versoOptsPanels) versoOptsPanels.style.display = "none"

        }
        if (mode === "local") {
            if (groqPanels)  groqPanels.style.display  = "none";
            if (localOptsPanels) localOptsPanels.style.display   = "block";
            if (localPanels) localPanels.style.display = "block";
            if (grokPanels)  grokPanels.style.display  = "none";
            if (versoPanels) versoPanels.style.display  = "none";
            if(versoOptsPanels) versoOptsPanels.style.display = "none"
        }
        if (mode === "verso") {
            if (groqPanels)  groqPanels.style.display  = "none";
            if (localOptsPanels) localOptsPanels.style.display   = "none";
            if (localPanels) localPanels.style.display = "none";
            if (grokPanels)  grokPanels.style.display  = "none";
            if (versoPanels) versoPanels.style.display  = "block";
        }
    });


    // Optional: remember last selection
    const savedMode = localStorage.getItem("modeSelector");
    if (savedMode) {
        modeSelector.value = savedMode;
        modeSelector.dispatchEvent(new Event("change"));
    }

    modeSelector.addEventListener("change", (e) => {
        localStorage.setItem("modeSelector", e.target.value);
    });

    modeSelector.addEventListener("change", (e) => {
        localStorage.setItem("modeSelector", e.target.value);
    });

function showToast(message) {
    const toast = document.createElement("div");
    toast.textContent = message;
    toast.className = "toast";
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 2000);
}
button.addEventListener("click", () => {
    const key = input.value.trim();
    if (!key) {
        showToast("Please enter a Groq key.");
        return;
    }

    // Example: store in localStorage
    localStorage.setItem("groqKey", key);

    // Or send to main process if Electron
    window.api.setGroqKey(key);
    showToast("Groq key saved!");
});

const storedKey = localStorage.getItem("groqKey");
if (storedKey) input.value = storedKey;


const bgChatContainer = document.getElementById('bg-chat');
document.addEventListener('change', (e) => {
if (!e.target.classList.contains('local-model-toggle')) return;

const container = e.target.closest('.agent-instance');
if (!container) return;

const groqSelect = container.querySelector('#groq-select');
const localSelect = container.querySelector('#local-select');

if (e.target.checked) {
    groqSelect.style.display = 'none';
    groqSelect.disabled = true;

    localSelect.style.display = '';
    localSelect.disabled = false;
} else {
    groqSelect.style.display = '';
    groqSelect.disabled = false;

    localSelect.style.display = 'none';
    localSelect.disabled = true;
}
});

// Optional: Initialize new agents with proper toggle
function initializeAgent(agentEl) {
    const checkbox = agentEl.querySelector('.local-model-toggle');
    if (!checkbox) return;

    // Trigger the toggle once to set initial state
    checkbox.dispatchEvent(new Event('change'));
}

// Example: spawning a new agent VISUALLY
function addAgent() {
    const template = document.getElementById('agent-template');
    const clone = template.content.cloneNode(true);
    const container = clone.querySelector('.agent-instance');

    document.getElementById('workflow-steps').appendChild(clone);
    initializeAgent(container);

    // After adding agent, update + Step button state
    updateAddStepButton();
    setAgentExists(true); // enable + Step once agent exists
}

// Enable / disable + Step button based on agent existence VISUALLY
function updateAddStepButton() {
    const agents = document.querySelectorAll('.agent-instance');
    addStepBtn.disabled = agents.length === 0;
}
let workflowStepCounter = 0;

function addWorkflowStep(){

    const template = document.getElementById("workflow-step-template");
    const container = document.getElementById("workflow-steps");

    const clone = template.content.cloneNode(true);

    const stepIndex = ++workflowStepCounter;

    const indexLabel = clone.querySelector(".workflow-index");
    indexLabel.textContent = stepIndex;

    const removeBtn = clone.querySelector(".workflow-remove");

    removeBtn.addEventListener("click", (e)=>{
        e.target.closest(".workflow-step").remove();
        reindexWorkflow();
    });

    populateAgentDropdown(clone);

    container.appendChild(clone);
}

function populateAgentDropdown(stepClone){

    const agentSelect = stepClone.querySelector(".workflow-agent");

    const agents = document.querySelectorAll(".agent-instance");

    agents.forEach(agent => {

        const nameInput = agent.querySelector(".agent-name");

        const option = document.createElement("option");
        option.value = nameInput.value || "Unnamed Agent";
        option.textContent = option.value;

        agentSelect.appendChild(option);
    });
}

function getAgentByName(name) {

    const agents = window.agents || [];

    return agents.find(a => a.name === name);
}

async function runAgent(agent, input) {
    const prompt = `
    ${agent.systemPrompt || ""}

    User Input:
    ${input}

    Respond as the agent.
    `;

    const response = await generateResponse(prompt, agent.model);
    return response;
}

// Add this in a <script> block or your JS file
async function executeWorkflow() {
const agentList = document.querySelectorAll('.agent-instance');
const agentsMap = {}; // Map agent names -> instance IDs

// 1️⃣ Create / spawn all agents
for (const agentEl of agentList) {
const name   = agentEl.querySelector('.agent-name').value;
const model  = agentEl.querySelector('.agent-model').value;
const prompt = agentEl.querySelector('.agent-prompt').value;

if (!name || !model) {
    console.warn('Skipping agent with missing name or model');
    continue;
}

// Spawn agent via engine
const agentInstance = await window.engine.spawnAgent({
    name,
    model,
    systemPrompt: prompt
});

agentsMap[name] = agentInstance.id; // Keep track of instance IDs
}

// 2️⃣ Read workflow steps
const steps = Array.from(document.querySelectorAll('.workflow-step')).map(step => {
const agentName = step.querySelector('.workflow-agent').value;
const action = step.querySelector('.workflow-action').value;
return { agentName, action };
});

// 3️⃣ Execute steps sequentially
for (const step of steps) {
const agentId = agentsMap[step.agentName];
if (!agentId) {
    console.warn('No agent found for step:', step);
    continue;
}

console.log(`Executing ${step.action} for agent ${step.agentName}`);

// Example: different actions
switch (step.action) {
    case 'respond':
        await window.engine.queryAgent(agentId, "Respond to workflow prompt", 1);
        break;
    case 'research':
        // Could call a RAG or Web search function
        await window.engine.engineIngestChats(agentId, ["Research files or data"]);
        break;
    case 'code':
        // Possibly generate code
        await window.engine.queryAgent(agentId, "Generate code snippet", 1);
        break;
    case 'tool':
        console.log('Tool action not implemented yet');
        break;
}
}

console.log('Workflow complete!');
}

// Attach to button
document.getElementById('start-pipeline-btn').onclick = executeWorkflow;


function reindexWorkflow(){

    const steps = document.querySelectorAll(".workflow-step");

    steps.forEach((step,i)=>{
        step.querySelector(".workflow-index").textContent = i+1;
    });

    workflowStepCounter = steps.length;
}

// --- 1. Navigation ---
function switchTab(tabId) {
    document.querySelectorAll('.module-section').forEach(el => el.classList.remove('active'));
    document.getElementById('module-' + tabId).classList.add('active');

    document.querySelectorAll('nav button').forEach(btn => {
        btn.classList.remove('active-nav');
        btn.classList.add('inactive-nav');
    });
    const activeBtn = document.getElementById('nav-' + tabId);
    activeBtn.classList.remove('inactive-nav');
    activeBtn.classList.add('active-nav');

    resizeCanvases();
}
document.addEventListener("DOMContentLoaded", () => {
    const neuralModule = document.getElementById("bg-chat");
        if (neuralModule) {
        // Make sure the section is visible before initializing
        setTimeout(() => {
            if (!window.currentBG) {
                window.currentBG = initMatrixRain();
            }
        }, 100); // small delay to let CSS render
    }
});

let agentCounter = 0;

    function switchBG(type) {
    // Clean previous
    var bg = undefined
    if (bg != undefined && currentBG?.cleanup) {
        currentBG.cleanup();
        currentBG = null;
    }
    const accent = getComputedStyle(document.documentElement)
        .getPropertyValue('--accent')
        .trim();
    switch (type) {
        case "Orbital View (3D)":
            currentBG = initChatBackground();
            currentBG.setAccent(accent);
            break;

        case "Neural Network (3D)":
            currentBG = initCircuitBackground();
            currentBG.setAccent(accent);
            break;

        case "Cold Rain":
            currentBG = initRainBackground();
            currentBG.setAccent(accent);
            break;

        case "Code Rain (Matrix)":
            currentBG = initMatrixRain();
            currentBG.setAccent(accent);
            break;

        case "Solid Black (Perf)":
            if (currentBG?.cleanup) {
                currentBG.cleanup();
                currentBG = null;
            }
            break;

        case "Custom":
            currentBG = initImageBackground(file);
    }
}

function loadSavedAppearance() {
    const savedTheme = localStorage.getItem('cybel-theme') || 'dark';
    const saved = localStorage.getItem("appearanceConfig");
    if (!saved){ 
        console.log("no saved theme");
        return;
    }

    const appearance = JSON.parse(saved);
    if (appearance.accent) {
        document.documentElement.style.setProperty('--accent', appearance.accent);
    }

    if (appearance.theme === "light") {
        document.documentElement.classList.add("light");
        document.documentElement.classList.remove("dark");
    } else {
        document.documentElement.classList.add("dark");
        document.documentElement.classList.remove("light");
    }

    if (appearance.background) {
        const bgSelect = document.getElementById("bgSelector");
        if (bgSelect) {
            bgSelect.value = appearance.background;
        }
        switchBG(appearance.background);
        setTheme(savedTheme);
        saveAppearance();
    }
}
function updateWorkflowDropdowns() {
const selects = document.querySelectorAll(".workflow-agent");
selects.forEach(sel => {
const option = document.createElement("option");
option.value = agentId;
option.textContent = agentConfig.id; // or a friendly name
sel.appendChild(option);
});
}

function createAgent() {
    const template = document.getElementById("agent-template");
    const agentList = document.getElementById("agent-list");
    const clone = template.content.cloneNode(true);
    const agentId = "agent_" + (++agentCounter);
    const agentRoot = clone.querySelector(".agent-instance");
    agentRoot.dataset.agentId = agentId;
    const modelSelect = clone.querySelector(".agent-model");
    const localToggle = clone.querySelector(".local-model-toggle");
    localToggle.addEventListener("change", () => {
    modelSelect.disabled = localToggle.checked;
    if(localToggle.checked){
        modelSelect.style.opacity = "0.5";
    } else {
        modelSelect.style.opacity = "1";
    }
    });

    agentList.appendChild(clone);
    setAgentExists(true); // enable + Step once agent exists

    const agentConfig = {
    id: agentId,
    provider: localToggle.checked ? "local" : modelSelect.value,
    tools: [], // later: read from Tools checkboxes
    mode: "active",
    embeddingDim: 1536, // default or configurable
    secretKey: null // optional for provider API key
    };

    // Spawn the instance in your InstanceEngine
    engine.spawn(agentConfig)
    .then(result => {
        if (!result.success) {
            alert("Agent creation failed: " + result.error);
        } else {
            console.log("Agent instance created:", result.instance);
        }
    });
}

const addStepBtn = document.querySelector('.workflow-add');

// Example: your agent state
let agentExists = false;

function setAgentExists(exists) {
agentExists = exists;
addStepBtn.disabled = !agentExists; // disable if no agent
}

// Optional: initialize button state on load
addStepBtn.disabled = !agentExists;

// --- Theme System ---
function setTheme(mode) {
    if (mode === 'light') {
        document.body.classList.add('light-theme');
    } else {
        document.body.classList.remove('light-theme');
    }
    localStorage.setItem('cybel-theme', mode);
}

function loadSavedTheme() {
    const saved = localStorage.getItem('cybel-theme') || 'dark';
    setTheme(saved);
}

// FIX: loadSavedAccent also updates globe after Three.js is ready
function loadSavedAccent() {
    const saved = localStorage.getItem('cybel-accent');
    if (!saved) return;
    document.documentElement.style.setProperty('--primary-accent', saved);
    document.documentElement.style.setProperty('--glass-border', saved + '55');

    document.querySelectorAll('.accent-color').forEach(el => {
        const color = el.getAttribute('onclick').match(/'(#[^']+)'/)?.[1];
        if (color === saved) el.classList.add('active');
    });
}
// --- Init ---
window.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    loadSavedAppearance();
    loadSavedTheme();
    loadSavedAccent();
    //initChatBackground();

    // After globe is built, apply saved accent to it
    const savedAccent = localStorage.getItem('cybel-accent');
    if (savedAccent) updateGlobeAccent();

    const dropZoneEl = document.getElementById('bg-drop-zone');
    dropZoneEl.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZoneEl.classList.add('border-cyan-400');
    });
    dropZoneEl.addEventListener('dragleave', () => {
        dropZoneEl.classList.remove('border-cyan-400');
    });
    dropZoneEl.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZoneEl.classList.remove('border-cyan-400');
        dropZoneEl.querySelector('p').innerText = "Image Loaded: custom_bg.png";
        dropZoneEl.querySelector('i').classList.add('text-green-400');
        dropZoneEl.querySelector('i').classList.remove('text-cyan-600');
    });
});
// Hook into module activation
function activateModule(id) {
    document.querySelectorAll('.module-section').forEach(s => s.classList.remove('active'));
    const section = document.getElementById(id);
    section.classList.add('active');
    if (id === 'bg-chat') {
        if (!currentBG) currentBG = initMatrixRain();
    }else if (id === 'module-management') {
        if (!currentBG) currentBG = initMatrixRain();
        currentBG.start(); // start rain **after module becomes visible**
    } else if (currentBG) {
        currentBG.cleanup(); // stop rain when switching away
        currentBG = null;
    }
}

// --- 2. Chat Background: Holographic Globe ---
function initMatrixRain() {

const canvas = document.getElementById('bg-chatB');
const div3D = document.getElementById('bg-chat');

if (!canvas) return;

div3D.classList.remove('active');
canvas.classList.add('active');

const ctx = canvas.getContext('2d');

const fontSize = 16;
const letters = 'アカサタナハマヤラワABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

let columns;
let drops;
let running = true;

function resize() {
canvas.width = canvas.parentElement.clientWidth;
canvas.height = canvas.parentElement.clientHeight;

columns = Math.floor(canvas.width / fontSize);
drops = Array(columns).fill(0);
}

resize();
window.addEventListener('resize', resize);

function draw() {

if (!running) return;

const light = document.body.classList.contains('light-theme');

const accent = getComputedStyle(document.documentElement)
    .getPropertyValue('--accent')
    .trim();

ctx.fillStyle = light
    ? 'rgba(240,240,240,0.1)'
    : 'rgba(0,0,0,0.05)';

ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.fillStyle = accent;
ctx.font = fontSize + 'px monospace';

for (let i = 0; i < drops.length; i++) {

    const text = letters[Math.floor(Math.random() * letters.length)];

    ctx.fillText(text, i * fontSize, drops[i] * fontSize);

    drops[i]++;

    if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
        drops[i] = 0;
    }
}

requestAnimationFrame(draw);
}

draw();

return {
cleanup() {
    running = false;
    window.removeEventListener('resize', resize);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.classList.remove('active');
}
};
}

function initImageBackground(imageSrc) {
    const container = document.getElementById('bg-chat');
    const scene     = new THREE.Scene();
    const camera    = new THREE.OrthographicCamera(
        -1, 1, 1, -1, 0.1, 10
    );
    camera.position.z = 1;

const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const loader = new THREE.TextureLoader();

const texture = loader.load(imageSrc, () => {
    renderer.render(scene, camera);
});

texture.minFilter = THREE.LinearFilter;
const geometry = new THREE.PlaneGeometry(2, 2);
const material = new THREE.MeshBasicMaterial({
    map: texture
});

const plane = new THREE.Mesh(geometry, material);
scene.add(plane);

function cleanup() {
    renderer.dispose();
    geometry.dispose();
    material.dispose();
    texture.dispose();
    container.removeChild(renderer.domElement);
}

function setAccent() {
    // Not needed for image bg, but keeps API consistent
}

renderer.render(scene, camera);

return {
    cleanup,
    setAccent
};
}

function initChatBackground() {
    const container = document.getElementById('bg-chat');
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.02);

    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 15;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Globe
    const globeGeometry = new THREE.IcosahedronGeometry(6, 2);
    const wireframeGeometry = new THREE.WireframeGeometry(globeGeometry);
    const accent = getComputedStyle(document.documentElement)
    .getPropertyValue('--accent')
    .trim();
    let accentColor = new THREE.Color(accent);
    const globeMaterial = new THREE.LineBasicMaterial({ color: accentColor, transparent: true, opacity: 0.15 });
    const globe = new THREE.LineSegments(wireframeGeometry, globeMaterial);
    scene.add(globe);

    // Core
    const coreGeo = new THREE.IcosahedronGeometry(2, 1);
    const coreMat = new THREE.MeshBasicMaterial({ color: 0x00ffff, wireframe: true, transparent: true, opacity: 0.3 });
    const core = new THREE.Mesh(coreGeo, coreMat);
    scene.add(core);

    // Particles
    const particlesGeo = new THREE.BufferGeometry();
    const particleCount = 400;
    const posArray = new Float32Array(particleCount * 3);
    for(let i = 0; i < particleCount * 3; i++) posArray[i] = (Math.random() - 0.5) * 25; 
    particlesGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    const particlesMat = new THREE.PointsMaterial({ size: 0.1, color: accentColor, transparent: true, opacity: 0.6 });
    const particles = new THREE.Points(particlesGeo, particlesMat);
    scene.add(particles);

    let animationId;
    const animate = () => {
        animationId = requestAnimationFrame(animate);
        globe.rotation.y += 0.002;
        globe.rotation.x += 0.0005;
        core.rotation.y -= 0.004;
        particles.rotation.y += 0.0005;
        renderer.render(scene, camera);
    };
animate();

// ---- EXPOSED METHODS ----
function setAccent(color) {
    accentColor = new THREE.Color(color);
    globeMaterial.color.set(accentColor);
    particlesMat.color.set(accentColor);
    globeMaterial.needsUpdate = true;
    particlesMat.needsUpdate = true;
}

function cleanup() {
    cancelAnimationFrame(animationId);
    renderer.dispose();
    globeGeometry.dispose();
    wireframeGeometry.dispose();
    particlesGeo.dispose();
    globeMaterial.dispose();
    particlesMat.dispose();
    container.removeChild(renderer.domElement);
}

// ---- HANDLE RESIZE ----
window.addEventListener('resize', () => {
    if (container.clientWidth > 0 && container.clientHeight > 0) {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }
});

return { cleanup, setAccent };
}

function initCircuitBackground() {
const container = document.getElementById('bg-chat');

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000000, 0.02);

const camera = new THREE.PerspectiveCamera(
75,
container.clientWidth / container.clientHeight,
0.1,
1000
);
camera.position.z = 20;

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

// ---- ACCENT COLOR ----
const accent = getComputedStyle(document.documentElement)
.getPropertyValue('--accent')
.trim();
let accentColor = new THREE.Color(accent);

// --- Nodes ---
const nodeGeometry = new THREE.SphereGeometry(0.2, 6, 6);
const nodeMaterial = new THREE.MeshBasicMaterial({ color: accentColor });
const nodes = [];
const nodeCount = 50;

for (let i = 0; i < nodeCount; i++) {
const node = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
node.position.set(
    (Math.random() - 0.5) * 30,
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 30
);
scene.add(node);
nodes.push(node);
}

// --- Connections (Lines) ---
const lineMaterial = new THREE.LineBasicMaterial({ color: accentColor, opacity: 0.2, transparent: true });
const lineGeometry = new THREE.BufferGeometry();
const positions = new Float32Array(nodeCount * nodeCount * 3 * 2); // max possible lines
lineGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const lineSegments = new THREE.LineSegments(lineGeometry, lineMaterial);
scene.add(lineSegments);

function updateConnections() {
let ptr = 0;
for (let i = 0; i < nodeCount; i++) {
    for (let j = i + 1; j < nodeCount; j++) {
        const dist = nodes[i].position.distanceTo(nodes[j].position);
        if (dist < 8) {
            positions[ptr++] = nodes[i].position.x;
            positions[ptr++] = nodes[i].position.y;
            positions[ptr++] = nodes[i].position.z;

            positions[ptr++] = nodes[j].position.x;
            positions[ptr++] = nodes[j].position.y;
            positions[ptr++] = nodes[j].position.z;
        }
    }
}
lineGeometry.setDrawRange(0, ptr / 3);
lineGeometry.attributes.position.needsUpdate = true;
}

// --- Animate ---
let animId;
function animate() {
animId = requestAnimationFrame(animate);

nodes.forEach(n => {
    n.rotation.x += 0.002;
    n.rotation.y += 0.003;
});

updateConnections();
renderer.render(scene, camera);
}
animate();

// --- Exposed Methods ---
function setAccent(color) {
accentColor = new THREE.Color(color);
nodeMaterial.color.set(accentColor);
lineMaterial.color.set(accentColor);
nodeMaterial.needsUpdate = true;
lineMaterial.needsUpdate = true;
}

function cleanup() {
cancelAnimationFrame(animId);
renderer.dispose();
nodeGeometry.dispose();
lineGeometry.dispose();
nodeMaterial.dispose();
lineMaterial.dispose();
container.removeChild(renderer.domElement);
}

window.addEventListener('resize', () => {
camera.aspect = container.clientWidth / container.clientHeight;
camera.updateProjectionMatrix();
renderer.setSize(container.clientWidth, container.clientHeight);
});

return { setAccent, cleanup };
}   

function initMatrixBotBackground() {
const canvas = document.getElementById('bg-chat');
const div3D = document.getElementById('bg-chatB');

if (!canvas) return;

div3D.classList.remove('active');
canvas.classList.add('active');

if (!canvas) return;
const ctx = canvas.getContext('2d');

const fontSize = 16;
const chars = '010101XYZAUTOMATSYSTEMアカサタナハマヤラワABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
let columns, drops;

function resize() {
canvas.width  = canvas.parentElement.clientWidth;
canvas.height = canvas.parentElement.clientHeight;
columns = Math.floor(canvas.width / fontSize);
drops = Array(columns).fill(0);
}
resize();
window.addEventListener('resize', resize);

function draw() {
const light = document.body.classList.contains('light-theme');
ctx.fillStyle = light ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.fillStyle = light ? '#009900' : '#0F0';
ctx.font = fontSize + 'px monospace';

for (let i = 0; i < drops.length; i++) {
    ctx.fillText(chars[Math.floor(Math.random() * chars.length)], i * fontSize, drops[i] * fontSize);
    if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
    drops[i]++;
}

requestAnimationFrame(draw);
}

draw();
}

function createCharSprite(char, color) {

const canvas = document.createElement("canvas");
canvas.width = 64;
canvas.height = 64;

const ctx = canvas.getContext("2d");
ctx.fillStyle = "transparent";
ctx.fillRect(0,0,64,64);

ctx.fillStyle = color;
ctx.font = "48px monospace";
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.fillText(char, 32, 32);

const texture = new THREE.CanvasTexture(canvas);

const material = new THREE.SpriteMaterial({
map: texture,
transparent: true
});

return new THREE.Sprite(material);
}

function initRainBackground() {
const container = document.getElementById('bg-chat'); // reuse bg-chat for consistency
const div3D     = document.getElementById('bg-chatB');

if (!canvas) return;

container.classList.add('active');
div3D.classList.remove('active');

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000000, 0.02);

const camera = new THREE.PerspectiveCamera(
75,
container.clientWidth / container.clientHeight,
0.1,
1000
);
camera.position.z = 20;

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

// ---- PARTICLES AS CHARACTERS ----
const chars = '010101XYZAUTOMATSYSTEMアカサタナハマヤラワABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
const particleCount = 1000;
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(particleCount * 3);
const speeds = new Float32Array(particleCount);

for (let i = 0; i < particleCount; i++) {
positions[i * 3 + 0] = (Math.random() - 0.5) * 50; // x
positions[i * 3 + 1] = Math.random() * 50;         // y
positions[i * 3 + 2] = (Math.random() - 0.5) * 50; // z
speeds[i] = 0.05 + Math.random() * 0.1;
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
const material = new THREE.PointsMaterial({
size: 0.5,
color: new THREE.Color(accent),
transparent: true,
opacity: 0.6
});

const points = new THREE.Points(geometry, material);
scene.add(points);

let animationId;
function animate() {
animationId = requestAnimationFrame(animate);
const pos = geometry.attributes.position.array;
for (let i = 0; i < particleCount; i++) {
    pos[i * 3 + 1] -= speeds[i]; // move down
    if (pos[i * 3 + 1] < -25) pos[i * 3 + 1] = 25; // loop
}
geometry.attributes.position.needsUpdate = true;
renderer.render(scene, camera);
}
animate();

// ---- METHODS ----
function setAccent(color) {
material.color.set(color);
material.needsUpdate = true;
}

function cleanup() {
cancelAnimationFrame(animationId);
renderer.dispose();
geometry.dispose();
material.dispose();
container.removeChild(renderer.domElement);
}

window.addEventListener('resize', () => {
if (container.clientWidth > 0) {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}
});

return { setAccent, cleanup };
}

// --- 4. UI Logic ---
function toggleSwitch(btn) {
    const knob = btn.querySelector('span');
    const isActive = btn.classList.contains('bg-green-600');
    if (isActive) {
        btn.classList.remove('bg-green-600'); btn.classList.add('bg-gray-700');
        knob.classList.remove('right-1'); knob.classList.add('left-1');
    } else {
        btn.classList.remove('bg-gray-700'); btn.classList.add('bg-green-600');
        knob.classList.remove('left-1'); knob.classList.add('right-1');
    }
}

function closeModal(id) {
    const modal = document.getElementById(id);
    modal.classList.add('opacity-0');
    modal.querySelector('.modal-content').classList.remove('scale-100');
    modal.querySelector('.modal-content').classList.add('scale-95');
    setTimeout(() => modal.classList.add('hidden'), 200);
}

function showModal(title, message) {
    const modal = document.getElementById('custom-modal');
    document.getElementById('modal-title').innerText = title;
    document.getElementById('modal-message').innerText = message;
    modal.classList.remove('hidden');
    setTimeout(() => {
        modal.classList.remove('opacity-0');
        modal.querySelector('.modal-content').classList.remove('scale-95');
        modal.querySelector('.modal-content').classList.add('scale-100');
    }, 10);
}

function saveConfiguration(){
    showModal('Config Saved', 'Parameters Reloaded.');
    const tempValue = document.getElementById('contextValue');
    const contextValue = document.getElementById('tempValue');
    const selector = document.getElementById("model-selector");
    window.api.updateEngine({ mode: selector.value });
    window.api.setTemperature(tempValue);
    window.api.setContextWindowKey(contextValue);
}

async function handleChatSubmit(e) {
    e.preventDefault();

    const input = document.getElementById('chat-input');
    const groqKeyInput = document.getElementById('groq-key-input');
    const history = document.getElementById('chat-history');
    const text    = input.value.trim();
    if (!text) return;

    /*const keyText = groqKeyInput.value.trim();*/
    /*if(keyText){
        await window.api.setGroqKey(text);
    }*/
    // 1. Show user message
    history.insertAdjacentHTML('beforeend', `
        <div class="flex gap-4 justify-end">
            <div class="text-accent/30 border border-accent/50 p-4 rounded-lg rounded-tr-none shadow-lg max-w-[80%]">
                <p class="text-accent text-sm leading-relaxed">${text}</p>
            </div>
        </div>
    `);

    input.value = '';
    history.scrollTop = history.scrollHeight;
    const avatarSrc = getCurrentThemeAvatar(); // e.g. './assets/avatar_red.png'

    // 2. Temporary "thinking" message
    const thinkingId = `thinking-${Date.now()}`;
    history.insertAdjacentHTML('beforeend', `
        <div id="${thinkingId}" class="flex gap-4">
            <div class="w-8 h-8 rounded-full overflow-hidden text-accent flex items-center justify-center border border-accent/30">
                <img
                            id="avatarImage"
                            src=""${avatarSrc}""
                            alt="AI Avatar"
                            class="w-full h-full object-cover"
                />
            </div>
            <div class="bg-accent/60 border border-accent/30 p-4 rounded-lg rounded-tl-none max-w-[80%]">
                <p class="text-accent text-sm font-mono animate-pulse">Processing…</p>
            </div>
        </div>
    `);
    lucide.createIcons();
    history.scrollTop = history.scrollHeight;
    try {
        // Send message via IPC
        const content = await window.api.sendMessage(text);

        if (!content) {
            throw new Error("No response from backend");
        }

        // Parse markdown
        const html = marked.parse(content);
        const avatarSrc = getCurrentThemeAvatar(); // e.g. './assets/avatar_red.png'

        // Replace thinking message with AI response
        document.getElementById(thinkingId).outerHTML = `
            <div class="flex gap-4">
                <div class="w-8 h-8 rounded-full overflow-hidden bg-accent/50 flex items-center justify-center border border-accent/30">
                    <img
                            id="avatarImage"
                            src="${avatarSrc}"
                            alt="AI Avatar"
                            class="w-full h-full object-cover"
                    />
                </div>
                <div class="bg-accent/60 border border-accent/30 p-4 rounded-lg rounded-tl-none shadow-lg max-w-[80%]">
                    <div class="markdown-content">
                        <p class="text-accent">${html}</p>
                    </div>
                </div>
            </div>
        `;
        lucide.createIcons();
        history.scrollTop = history.scrollHeight;

    } catch (err) {
        document.getElementById(thinkingId).remove();
        showModal("Connection Error", "Unable to reach CYBEL core." + err);
    }
}

function handleFileUpload(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        const history = document.getElementById('chat-history');
        
        const fileMsgHTML = `
        <div class="flex gap-4 justify-end">
            <div class="bg-accent/30 border border-accent/50 p-4 rounded-lg rounded-tr-none shadow-lg max-w-[80%]">
                <p class="text-accent text-sm leading-relaxed flex items-center gap-2">
                    <i data-lucide="file"></i> Attached: ${file.name}
                </p>
            </div>
        </div>`;
        
        history.insertAdjacentHTML('beforeend', fileMsgHTML);
        history.scrollTop = history.scrollHeight;
        lucide.createIcons();
    }
}

// --- 6. Create Bot Logic ---
let selectedBotType = null;
function saveAppearance() {
    const bgSelect = document.getElementById('bgSelector');
    const appearance = {
        accent: getComputedStyle(document.documentElement)
                    .getPropertyValue('--accent')
                    .trim(),
        theme: document.documentElement.dataset.theme,
        background: bgSelect.value
    };

    localStorage.setItem("appearanceConfig", JSON.stringify(appearance));
}

function openCreateBotModal() {
    document.getElementById('create-bot-modal').classList.remove('hidden');
    setTimeout(() => {
        document.getElementById('create-bot-modal').classList.remove('opacity-0');
        document.getElementById('create-bot-modal').querySelector('.modal-content').classList.remove('scale-95');
        document.getElementById('create-bot-modal').querySelector('.modal-content').classList.add('scale-100');
    }, 10);
}

function selectBotType(type) {
    selectedBotType = type;
    document.getElementById('type-character').classList.remove('selected');
    document.getElementById('type-agent').classList.remove('selected');
    document.getElementById('type-' + type).classList.add('selected');
    window.api.updateCharacter(selectedBotType);
}

function confirmCreateBot() {
    const name = document.getElementById('new-bot-name').value;
    if (!name) { alert("Please enter an Instance Designation."); return; }
    if (!selectedBotType) { alert("Please select a Cognitive Architecture."); return; }
    closeModal('create-bot-modal');
    showModal('Initializing Instance', `Instance <strong>${name}</strong> (${selectedBotType.toUpperCase()} mode) is being provisioned...`);
}

// --- 7. Creative Mode Logic (Restored) ---
let creativeStep = 1;

function setCreativeInput(type) {
    creativeStep = 2;
    updateCreativeView();
}

function setCreativeOutput(type) {
    creativeStep = 3;
    updateCreativeView();
}

function prevCreativeStep() {
    if (creativeStep > 1) creativeStep--;
    updateCreativeView();
}

function updateCreativeView() {
    document.getElementById('creative-step-1').classList.add('hidden');
    document.getElementById('creative-step-2').classList.add('hidden');
    document.getElementById('creative-step-3').classList.add('hidden');
    
    document.getElementById(`creative-step-${creativeStep}`).classList.remove('hidden');

    const ind2 = document.getElementById('step-ind-2');
    const ind3 = document.getElementById('step-ind-3');
    
    if (creativeStep >= 2) {
        ind2.classList.remove('bg-gray-800', 'text-gray-400');
        ind2.classList.add('bg-accent', 'text-black');
    } else {
        ind2.classList.add('bg-gray-800', 'text-gray-400');
        ind2.classList.remove('bg-accent', 'text-black');
    }

    if (creativeStep >= 3) {
        ind3.classList.remove('bg-gray-800', 'text-gray-400');
        ind3.classList.add('bg-accent', 'text-black');
    } else {
        ind3.classList.add('bg-gray-800', 'text-gray-400');
        ind3.classList.remove('bg-accent', 'text-black');
    }
}

function executeCreative() {
    showModal('Generation Initiated', 'The engine has received the parameters.');
    creativeStep = 1;
    updateCreativeView();
}

// --- 8. Engine Logic (Restored) ---
function runBenchmark() {
    const btn = event.currentTarget;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i data-lucide="loader" class="w-4 h-4 animate-spin"></i> RUNNING...';
    lucide.createIcons();
    setTimeout(() => {
        btn.innerHTML = originalText;
        lucide.createIcons();
        showModal('Benchmark Complete', 'Performance: 45.2 tokens/sec.');
    }, 2000);
}

// --- 9. Stats Loop & Helpers ---
setInterval(() => {
    const tempEl = document.getElementById('stat-temp');
    if(tempEl) {
        let t = parseFloat(tempEl.innerText) + (Math.random() * 0.4 - 0.2);
        tempEl.innerText = t.toFixed(1) + "°C";
        tempEl.className = t > 45 ? "font-mono text-red-500" : "font-mono text-green-400";
    }
    const latEl = document.getElementById('stat-latency');
    if(latEl) {
        let l = 3 + Math.random() * 4;
        latEl.innerText = l.toFixed(1) + "ms";
    }
}, 2000);

function resizeCanvases() {
    window.dispatchEvent(new Event('resize'));
}

let avatar = null;
function setCurrentThemeAvatar(av){
    avatar = av;
}
function getCurrentThemeAvatar() {
// Default avatar and color
let avatar = "./assets/avatar_blue.png";
let color = "#06b6d4"; // default accent color

const bgSelectS = document.getElementById('bgSelector');
if (bgSelectS && bgSelectS.value) {
color = bgSelectS.value;
    // Map of accent colors to avatars
const avatarMap = {
"#06b6d4": "./assets/avatar_blue.png",
"#3b82f6": "./assets/avatar_blue.png",
"#f59e0b": "./assets/avatar_gold.png",
"#22c55e": "./assets/avatar_green.png",
"#b34639": "./assets/avatar_red.png",
"#3C8E38": "./assets/avatar_emerald.png",
"#7E4D5D": "./assets/avatar_rose.png",
"#B87232": "./assets/avatar_gold.png",
"#5F268D": "./assets/avatar_logicgate.png",
"#627A5B": "./assets/avatar_emerald.png",
"#914D79": "./assets/avatar_pink.png",
"#9E8850": "./assets/avatar_yellow.png",
"#712925": "./assets/avatar_ultron.png",
"#2B2C2B": "./assets/avatar_black.png"
};

// Pick avatar based on color, fallback to default
if (avatarMap[color]) {
avatar = avatarMap[color];
}

return avatar;
}
return null;
}
const canvas = document.getElementById('bg-bot');
const ctx = canvas.getContext('2d');

// 1. Set actual canvas size (not CSS)
canvas.width = canvas.offsetWidth;
canvas.height = canvas.offsetHeight;

// 2. Font & fill style
ctx.font = '16px monospace';
ctx.fillStyle = 'lime';

// 3. Initialize drops AFTER width is set
const fontSize = 16;
const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
const columns = Math.floor(canvas.width / fontSize);
const drops = Array(columns).fill(0);

// 4. Draw loop
function draw() {
ctx.fillStyle = 'rgba(0,0,0,0.05)'; // fade
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.fillStyle = 'lime';
for (let i = 0; i < drops.length; i++) {
const text = letters[Math.floor(Math.random() * letters.length)];
ctx.fillText(text, i * fontSize, drops[i] * fontSize);

drops[i] = (drops[i] * fontSize > canvas.height && Math.random() > 0.975) ? 0 : drops[i] + 1;
}
}

setInterval(draw, 50);
// Init
window.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
        // Mode selector
    const modeSelector = document.getElementById("modeSelector");
    document.getElementById('hf-login').addEventListener('click', () => hub.loginHuggingFace());
    document.getElementById('ms-login').addEventListener('click', () => hub.loginModelScope());
    document.getElementById('login-token').addEventListener('click', () => hub.login);

    if (modeSelector) {
    modeSelector.addEventListener("change", (e) => {
        window.api.setMode(e.target.value);
    });
    }
    
            const themeButtons = document.querySelectorAll('[data-theme]');
                themeButtons.forEach(btn => {
                btn.addEventListener('click', () => {
                    // Remove selected from all
                    themeButtons.forEach(b => b.classList.remove('selected', 'bg-accent', 'text-white'));
                    
                    // Add selected styling
                    btn.classList.add('selected', 'bg-accent', 'text-white');

                    if (btn.dataset.theme === "dark") {
                        document.documentElement.classList.remove("light");
                        document.documentElement.classList.add("dark");
                        document.body.dataset.theme = "dark";
                    } else {
                        document.documentElement.classList.remove("dark");
                        document.documentElement.classList.add("light");
                        document.body.dataset.theme = "light";
                    }
                });
            });

            // Accent color swatches
            const colorSwatches = document.querySelectorAll('[data-color]');
                colorSwatches.forEach(swatch => {
                swatch.addEventListener('click', () => {
                    const color = swatch.dataset.color;
                    document.documentElement.style.setProperty('--accent', color);

                    // Remove selected from all
                    colorSwatches.forEach(s => s.classList.remove('selected', 'ring-4', 'ring-white'));
                    // Add selected to clicked
                    swatch.classList.add('selected', 'ring-4', 'ring-white');
                });
            });
    
            document.querySelectorAll('.SaveLoadOption').forEach(btn => {
            btn.addEventListener('click', async () => {
                const title = 'MyDoc'; // Replace with your selected document
                let res;

                switch (btn.innerText) {
                case 'SAVE':
                    res = await window.api.saveDocument({ title, content: 'Hello World' });
                    break;
                case 'LOAD':
                    res = await window.api.loadDocument(title);
                    break;
                case 'DELETE':
                    res = await window.api.deleteDocument(title);
                    break;
                case 'REPLICATE':
                    let t = title + "_copy";
                    res = await window.api.replicateDocument(t);
                    break;
                case 'MERGE':
                    res = await window.api.mergeDocument(
                    { title, content: 'Doc1 content' },
                    { title: title + '_2', content: 'Doc2 content' },
                    title + '_merged'
                    );
                    break;
                case 'EXPORT':
                    res = await window.api.exportDocument({ title, content: 'Hello' }, 'C:\\Users\\User\\Desktop\\export.json');
                    break;
                }

                console.log(res);
            });
            });


                //globeMaterial.color.set(color);
            const bgSelect = document.getElementById('bgSelector');
            function switchBG(type) {
                // Clean previous
                if (currentBG?.cleanup) {
                    currentBG.cleanup();
                    currentBG = null;
                }
                const accent = getComputedStyle(document.documentElement)
                    .getPropertyValue('--accent')
                    .trim();
                switch (type) {
                    case "Orbital View (3D)":
                        currentBG = initChatBackground();
                        currentBG.setAccent(accent);
                        break;

                    case "Neural Network (3D)":
                        currentBG = initCircuitBackground();
                        currentBG.setAccent(accent);
                        break;

                    case "Matrix Rain (2d)":
                        currentBG = initMatrixRain();
                        setBackground("2d");
                        break;
                        
                    case "Cold Rain (3d)":
                        currentBG = initRainBackground();
                        currentBG.setAccent(accent);
                        break;

                    case "Solid Black (Perf)":
                        if (currentBG?.cleanup) {
                            currentBG.cleanup();
                            currentBG = null;
                        }
                        break;

                    case "Custom":
                        currentBG = initImageBackground(file);
                }
            }

            const avatarMap = {
                "#06b6d4": "./assets/avatar_blue.png",
                "#3b82f6": "./assets/avatar_blue.png",
                "#f59e0b": "./assets/avatar_gold.png",
                "#22c55e": "./assets/avatar_green.png",
                "#b34639": "./assets/avatar_red.png",
                "#3C8E38": "./assets/avatar_emerald.png",
                "#7E4D5D": "./assets/avatar_rose.png",
                "#B87232": "./assets/avatar_gold.png",
                "#5F268D": "./assets/avatar_logicgate.png",
                "#627A5B": "./assets/avatar_emerald.png",
                "#914D79": "./assets/avatar_pink.png",
                "#9E8850": "./assets/avatar_yellow.png",
                "#712925": "./assets/avatar_ultron.png",
                "#2B2C2B": "./assets/avatar_black.png"
            };
            saveAppearance();

            const avatarImage = document.getElementById("avatarImage");

            colorSwatches.forEach(swatch => {
            swatch.addEventListener("click", () => {
                    const color = swatch.dataset.color;

                    document.documentElement.style.setProperty("--accent", color);
                    const bgSelectS = document.getElementById('bgSelector');
                    switchBG(bgSelectS.value)
                    // swap avatar
                    if (avatarMap[color]) {
                    avatarImage.src = avatarMap[color];
                    avatar = avatarMap[color];
                    }

                    // update selected ring
                    colorSwatches.forEach(s => 
                    s.classList.remove("ring-4", "ring-white")
                    );
                    swatch.classList.add("ring-4", "ring-white");
                });
            });

            bgSelect.addEventListener('change', e => switchBG(e.target.value));
                // Temperature slider
                const tempRange = document.getElementById('tempRange');
                const tempValue = document.getElementById('tempValue');
                if (tempRange && tempValue) {
                tempRange.addEventListener('input', () => {
                    tempValue.textContent = tempRange.value;
                });
            }

            // Context window slider
            const contextRange = document.getElementById('contextRange');
            const contextValue = document.getElementById('contextValue');
            if (contextRange && contextValue) {
                contextRange.addEventListener('input', () => {
                    contextValue.textContent = contextRange.value;
                });
            }
        });

    function handleImageFile(file) {
        if (!file || !file.type.startsWith("image/")) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            if (currentBG?.cleanup) currentBG.cleanup();
            currentBG = initImageBackground(event.target.result);
        };
        reader.readAsDataURL(file);
    }

    const selector = document.getElementById("modeSelector");
    selector.addEventListener("change", (e) => {
        window.api.updateEngine({ mode: selector.value });
    });

    const cselector = document.getElementById("character-selector");
    selector.addEventListener("change", (e) => {
        window.api.updateCharacter({ mode: selector.value });
    });
    // Drag and drop for appearance
    const dropZone  = document.getElementById('bg-drop-zone');
    const fileInput = document.getElementById('bg-file-input');

    // Click = open picker
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // File picker
    fileInput.addEventListener('change', (e) => {
        handleImageFile(e.target.files[0]);
    });
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-accent');
    });
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-accent');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-accent/20');

        const file = e.dataTransfer.files[0];
        if (!file || !file.type.startsWith("image/")) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            switchBG("Custom");
            currentBG = initImageBackground(event.target.result);
            saveAppearance();
        };
        reader.readAsDataURL(file);
    });
          
    
    async function executeWorkflow() {

        console.log("⚙️ Starting workflow...");

        const steps = document.querySelectorAll(".workflow-step");

        if (steps.length === 0) {
            console.warn("No workflow steps defined.");
            return;
        }

        let previousOutput = "";

        for (let i = 0; i < steps.length; i++) {

            const step = steps[i];

            const agentName = step.querySelector(".workflow-agent").value;
            const action = step.querySelector(".workflow-action").value;

            console.log(`Running Step ${i + 1}:`, agentName, action);

            if (!agentName) {
                console.warn("Step skipped: no agent selected");
                continue;
            }

            const agent = getAgentByName(agentName);

            if (!agent) {
                console.warn("Agent not found:", agentName);
                continue;
            }

            let result = "";

            switch (action) {

                case "respond":
                    result = await runAgent(agent, previousOutput);
                    break;

                case "research":
                    result = await runResearchTool(previousOutput);
                    break;

                case "code":
                    result = await runAgent(agent, "Write code for: " + previousOutput);
                    break;

                case "tool":
                    result = await runAgentTool(agent, previousOutput);
                    break;

            }

            previousOutput = result;

            console.log(`Step ${i + 1} output:`, result);
        }

        console.log("✅ Workflow finished.");
    }

// =====================================================
// CUSTOM BG FILE INPUT
// =====================================================
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        //if (animateId) cancelAnimationFrame(animateId);
        //if (renderer && renderer.domElement.parentNode) { renderer.domElement.parentNode.removeChild(renderer.domElement); renderer.dispose(); renderer = null; }
        bgChatContainer.style.backgroundImage = `url(${e.target.result})`;
        bgChatContainer.style.backgroundSize = 'cover';
        bgChatContainer.style.backgroundPosition = 'center';
        customBgSelected = true;
        dropZone.querySelector('p').innerText = `Loaded: ${file.name}`;
    };
    reader.readAsDataURL(file);
});

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-accent/80'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-accent/80'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-accent/80');
    dropZone.querySelector('p').innerText = "Image Loaded: custom_bg.png";
    dropZone.querySelector('i').classList.add('text-green-400');
    dropZone.querySelector('i').classList.remove('text-accent');
});

document.getElementById('remove-bg-btn').addEventListener('click', (() => {bgChatContainer.style.backgroundImage = null; dropZone.querySelector('p').innerText = `Ready for Upload`;}));
const downloadButton = document.getElementById("model_confirmation")
const modelInput     = document.getElementById("model-search-input")
const hfCheckbox = document.getElementById("source-hf")
const msCheckbox = document.getElementById("source-ms")
const modelDropdown  = document.getElementById("local-model-selector")

downloadButton.addEventListener("click", async () => {
    const dropdownName = modelDropdown?.value
    const modelName = modelInput.length > 0 ? modelInput : dropdownName
    if (!modelName) {
        alert("Enter a model name or choose one from the dropdown")
        return
    }

    const sources = {
        huggingface: hfCheckbox.checked,
        modelscope:  msCheckbox.checked
    }

    try {
        if (sources.huggingface) {
            downloadButton.disabled = true
            downloadButton.innerText = "Downloading..."
            let useOllama  = document.getElementById("modeSelector") === "Verso"
            await hub.downloadModel(modelName, {
                huggingface: hfCheckbox.checked,
                modelscope:  msCheckbox.checked,
                ollama:     useOllama
            })

            downloadButton.disabled = false
            downloadButton.innerText = "Downloaded Model"
        }

        if (sources.modelscope) {
            downloadButton.disabled = true
            downloadButton.innerText = "Downloading..."
            await hub.downloadFromModelScope(modelName)
            downloadButton.disabled = false
            downloadButton.innerText = "Downloaded Model"
        }
    } catch (err) {
        console.error("Model download failed:", err)
    }
})

window.createAgent      = createAgent;
window.addWorkflowStep  = addWorkflowStep;
window.executeWorkflow  = executeWorkflow;
window.closeModal       = closeModal;
window.switchTab        = switchTab;
window.handleChatSubmit = handleChatSubmit;