// renderer.js
import { AutoTokenizer, AutoModelForCausalLM } from "https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js"; //"@xenova/transformers";

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

function initModelSelects() {
    const container = document.querySelector(".agent-instance");
    if (!container) return;

    const map = {
        "groq-model-toggle":  "#groq-select",
        "local-model-toggle": ".local-select",
        "grok-model-toggle":  ".grok-select",
        "verso-model-toggle": ".verso-select"
    };

    // hide all selects
    Object.values(map).forEach(sel => {
        const el = container.querySelector(sel);
        if (!el) return;
        el.style.display = "none";
        el.disabled = true;
    });

    const checked = container.querySelector("input[type='radio']:checked");
    if (!checked) return;

    const selectId = map[[...checked.classList].find(c => map[c])];
    const selectEl = container.querySelector(selectId);

    if (selectEl) {
        selectEl.style.display = "block";
        selectEl.disabled = false;
    }
}


    // -------------------------------
    // Local Model Selector Manager
    // -------------------------------
    document.querySelector("#saveBtn").addEventListener("click", async () => {
        const files = await window.api.saveDocument(selectedSlot);
        console.log(files);
    });

    document.querySelector("#loadBtn").addEventListener("click", async () => {
        const files = await window.api.loadDocument();
        return files;
    });

    document.querySelector("#mergeBtn").addEventListener("click", async () => {
        resA  = await window.api.loadDocument();
        resB  = await window.api.loadDocument();
        if (!confirm("Terminate this agent instance?")) return;
        const files = await window.api.mergeDocument({baseTitle: resA, mergeTitle: resB, outputTitle: (resA.name + "_output")});
        console.log(files);
    });

    //document.getElementById("#delete-agent-btn").addEventListener("click", () => {
        //    window.api.deleteAgent();
        //});

    const localSelector  = document.getElementById("local-model-selector");
    const localSelectorV = document.getElementById("vs-model-selector");

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

// Initialize dropdown: mark installed models green, others red
    async function initVSLocalSelector() {
        installedModels =  []
        
        // TBA: make Installed Models call this and apply the results 
        //await modelHub.getLocalModels(); // ["Mistral 7b", "dolphin3", ...]
        Array.from(localSelectorV.options).forEach(option => {
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
 
    const input  = document.getElementById("groq-key-input");
    const button = document.getElementById("groq-key-confirm");
    const buttonGK = document.getElementById("grok-key-confirm");
    const buttonVS = document.getElementById("verso-key-confirm");

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
            localStorage.setItem("modeSelector", "groq");
        }
        if(mode === "grok"){
            if (groqPanels)  groqPanels.style.display  = "none";
            if (localOptsPanels) localOptsPanels.style.display   = "none";
            if (localPanels) localPanels.style.display = "none";
            if (grokPanels)  grokPanels.style.display  = "block";
            if (versoPanels) versoPanels.style.display  = "none";
            if(versoOptsPanels) versoOptsPanels.style.display = "none"
            localStorage.setItem("modeSelector", "grok");
        }
        if (mode === "local") {
            if (groqPanels)  groqPanels.style.display  = "none";
            if (localOptsPanels) localOptsPanels.style.display   = "block";
            if (localPanels) localPanels.style.display = "block";
            if (grokPanels)  grokPanels.style.display  = "none";
            if (versoPanels) versoPanels.style.display  = "none";
            if(versoOptsPanels) versoOptsPanels.style.display = "none"
            localStorage.setItem("modeSelector", "local");
        }
        if (mode === "verso") {
            if (groqPanels)  groqPanels.style.display  = "none";
            if (localOptsPanels) localOptsPanels.style.display   = "none";
            if (localPanels) localPanels.style.display = "none";
            if (grokPanels)  grokPanels.style.display  = "none";
            if (versoPanels) versoPanels.style.display  = "block";
            if(versoOptsPanels) versoOptsPanels.style.display = "block"
            localStorage.setItem("modeSelector", "verso");
        }
        localStorage.setItem("modeSelector", e.target.value);
        window.api.setGenerationMode(e.target.value);
    });

    modeSelector.dispatchEvent(new Event('change'));
    initLocalSelector();
    initVSLocalSelector();

    // Optional: remember last selection
    const savedMode = localStorage.getItem("modeSelector");
    if (savedMode) {
        modeSelector.value = savedMode;
        modeSelector.dispatchEvent(new Event("change"));
    }

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
        window.api.setTokenKey(key);
    });

    buttonGK.addEventListener("click", () => {
        const key = input.value.trim();
        if (!key) {
            showToast("Please enter a Grok key.");
            return;
        }
        window.api.setTokenKey(key);
    });

    buttonVS.addEventListener("click", () => {
        const key = input.value.trim();
        if (!key) {
            showToast("Please enter a Verso key.");
            return;
        }
        window.api.setTokenKey(key);
    });

    button.addEventListener("click", () => {
    const key = input.value.trim();
    if (!key) {
        showToast("Please enter a Groq key.");
        return;
    }
    // Example: store in localStorage
    localStorage.setItem("groqKey", key);

    // Or send to main process if Electron
    window.api.setTokenKey(key);
    showToast("Token key saved!");
});

const storedKey = localStorage.getItem("tokenKey");
if (storedKey) input.value = storedKey;


const bgChatContainer = document.getElementById('bg-chat');
document.addEventListener('change', (e) => {
    const target = e.target;
    if (!target.classList.contains('local-model-toggle') &&
        !target.classList.contains('grok-model-toggle') &&
        !target.classList.contains('verso-model-toggle')) return;

    const container = target.closest('.agent-instance');
    if (!container) return;

    const groqSelect = container.querySelector('#groq-select');
    const localSelect = container.querySelector('#local-select');
    const grokSelect = container.querySelector('#grok-select');
    const versoSelect = container.querySelector('#verso-select');

    // Hide everything initially
    [groqSelect, localSelect, grokSelect, versoSelect].forEach(el => {
        if (!el) return;
        el.style.display = 'none';
        el.disabled = true;
    });

    // Show the selected toggle’s select, or default to #groq-select if none
    if (container.querySelector('.local-model-toggle:checked')) {
        localSelect.style.display = 'block';
        localSelect.disabled = false;
    } else if (container.querySelector('.grok-model-toggle:checked')) {
        grokSelect.style.display = 'block';
        grokSelect.disabled = false;
    } else if (container.querySelector('.verso-model-toggle:checked')) {
        versoSelect.style.display = 'block';
        versoSelect.disabled = false;
    } else {
        groqSelect.style.display = 'block';
        groqSelect.disabled = false;
    }
});

// Optional: Initialize new agents with proper toggle
function initializeAgent(agentEl) {
    const checkbox  = agentEl.querySelector('.local-model-toggle');
    const checkboxG = agentEl.querySelector('.grok-model-toggle');
    const checkboxV = agentEl.querySelector('.verso-model-toggle');
    const groqSelect = agentEl.querySelector('#groq-select');
    initModelSelects();
    if (!checkbox || !checkboxG || !checkboxV || groqSelect.value === "None") return;

    // Trigger the toggle once to set initial state
    checkbox.dispatchEvent(new Event('change'));
    checkboxG.dispatchEvent(new Event('change'));
    checkboxV.dispatchEvent(new Event('change'));
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

    const response = await window.api.generateAgentResponse(agent.id, agent.model, prompt);
    return response;
}

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

async function createAgent(config = {}) {
    let engine      = window.api.getEngineInstance();
    const template  = document.getElementById("agent-template");
    const agentList = document.getElementById("agent-list");

    // Clone the template
    const clone = template.content.cloneNode(true);
    const agentRoot = clone.querySelector(".agent-instance");

    // Assign unique agent ID
    const agentId = "agent_" + (++agentCounter);
    agentRoot.dataset.agentId = agentId;

    // Set default values or override from config
    agentRoot.querySelector(".agent-name").value = config.name || `Agent ${agentCounter}`;
    agentRoot.querySelector(".agent-prompt").value = config.prompt || "";
    agentRoot.querySelector(".agent-title").textContent = config.title || "Agent Configuration";

    // Append to DOM
    agentList.appendChild(clone);

    // Hook up buttons dynamically
    const buttons = agentRoot.querySelectorAll(".config-buttons button");

    buttons.forEach(btn => {
        if (btn.textContent.includes("Import")) {
            btn.addEventListener("click", () => window.api.loadConfig(agentId));
        } else if (btn.textContent.includes("Export")) {
            btn.addEventListener("click", () => window.api.saveConfig(agentId));
        } else if (btn.textContent.includes("Delete")) {
            btn.addEventListener("click", () => deleteAgent(agentId));
        }
    });

    // Model toggles
    const localToggle = agentRoot.querySelector(".local-model-toggle");
    const modelSelects = agentRoot.querySelectorAll(".agent-model");

    localToggle.addEventListener("change", () => {
        modelSelects.forEach(select => {
            if (select.id === "local-select") {
                select.disabled = !localToggle.checked;
                select.style.display = localToggle.checked ? "inline-block" : "none";
            } else {
                select.disabled = localToggle.checked;
            }
        });
    });

    // Spawn agent in engine
    engine.spawn({
        id: agentId,
        provider: config.provider || "local",
        mode: "active",
        tools: [],
        embeddingDim: 1536
    }).then(result => {
        if (!result.success) alert("Agent creation failed: " + result.error);
        else console.log("Agent created:", agentId);
    });
}

function deleteAgent(agentId) {
  const agentRoot = document.querySelector(`[data-agent-id="${agentId}"]`);
  if (!agentRoot) {
    console.error("Agent element not found for ID:", agentId);
    return;
  }

  const confirmed = confirm("Terminate this agent?");
  if (!confirmed) return;

  agentRoot.remove();
  engine.destroy(agentId); // also tell backend
}

async function deleteBot(agentId, agentRoot) {
    let engine = window.api.getEngineInstance();
    const confirmed = confirm("Terminate this bot instance?");
    if (!confirmed) return;

    try {
        if (engine && engine.destroy) {
            await engine.destroy(agentId);
        }

        agentRoot.remove();
        console.log("Bot removed:", agentId);

        if (document.querySelectorAll(".agent-instance").length === 0) {
            setAgentExists(false);
        }

    } catch (err) {
        console.error("Bot deletion failed:", err);
        alert("Failed to terminate Bot.");

    }
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

/*document.querySelector("#ingestBtn").addEventListener("click", () => {
    window.api.openGDDialog();
});*/



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


async function handleChatSubmit(e) {
    e.preventDefault();

    const input = document.getElementById('chat-input');
    const groqKeyInput = document.getElementById('groq-key-input');
    const history = document.getElementById('chat-history');
    const text    = input.value.trim();
    if (!text) return;

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
                            src="${avatarSrc}"
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
        if (!window.api || !window.api.sendMessage) {
            throw new Error("CYBEL API bridge not available");
        }

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
let selectedSlot = null;

function setSelectedSlot(id) {
    selectedSlot = id;
    document.getElementById("selected-bot").textContent = selectedSlot;
}

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

function openCreateBotModal(id) {
    selectedSlot = id;

    const modal = document.getElementById('create-bot-modal');
    modal.classList.remove('hidden');

    setTimeout(() => {
        modal.classList.remove('opacity-0');
        modal.querySelector('.modal-content').classList.remove('scale-95');
        modal.querySelector('.modal-content').classList.add('scale-100');
    }, 10);
}


function renderBotCard(bot) {

    const slot = document.getElementById(bot.id);
    slot.className = "py-3 px-4 border border-accent/50 card glass-panel p-4";
    slot.innerHTML = `
        <div class="card-header">
            <div class="bot-icon">🤖</div>
            <div class="status-dot"></div>
        </div>

        <div class="agent-tag">${bot.type.toUpperCase()}</div>
        <div class="bot-name">${bot.name}|${document.getElementById('character-selector').value}</div>

        <div class="bot-desc">
            Instance initialized. Awaiting directive input.
        </div>

        <div class="card-footer">
            <button onclick="switchTab('chat')" class="icon-btn">></button>
            <!--<button class="icon-btn">📋</button>-->
            <button onclick="deleteBot('${bot.id}', this.parentElement.parentElement)" class="icon-btn">🗑️</button>
            <button onclick="openCreateBotModal('${bot.id}')" class="launch-btn">CONFIG</button>
        </div>
    `;
}

async function createBotInstance(name, type) {

    const botData = {
        id: selectedSlot,
        name: name,
        type: type,
        description: "New instance awaiting initialization..."
    };

    if (window.api && window.api.createBot) {
        await window.api.createBot(botData);
    }

    renderBotCard(botData);
}

function selectBotType(type) {
    selectedBotType = type;
    document.getElementById('type-character').classList.remove('selected');
    document.getElementById('type-utility').classList.remove('selected');
    document.getElementById('type-' + type).classList.add('selected');
    window.api.updateCharacter(selectedBotType);
}

function confirmCreateBot() {

    const name = document.getElementById('new-bot-name').value;

    if (!name) {
        alert("Please enter an Instance Designation.");
        return;
    }

    if (!selectedBotType) {
        alert("Please select a Cognitive Architecture.");
        return;
    }

    closeModal('create-bot-modal');

    showModal(
        'Initializing Instance',
        `Instance <strong>${name}</strong> (${selectedBotType.toUpperCase()} mode) is being provisioned...`
    );

    createBotInstance(name, selectedBotType);
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

    function reindexWorkflow(){
        const steps = document.querySelectorAll(".workflow-step");
        steps.forEach((step,i)=>{
            step.querySelector(".workflow-index").textContent = i+1;
        });
        workflowStepCounter = steps.length;
    }

    async function downloadHF(modelName){
        await AutoTokenizer.from_pretrained(modelName);
        await AutoModelForCausalLM.from_pretrained(modelName);
    }

    function downloadModelScope(modelName){
        return new Promise((resolve,reject)=>{

            const proc = spawn("modelscope", [
                "download",
                modelName
            ]);

            proc.on("close", code=>{
                if(code === 0) resolve();
                else reject("ModelScope download failed");
            });

        });
    }

    function downloadOllama(modelName) {
        return new Promise((resolve, reject) => {
            const proc = spawn("ollama", ["pull", modelName]);
            proc.stdout.on("data", d => {
                console.log(d.toString());
            });

            proc.stderr.on("data", d => {
                console.error(d.toString());
            });

            proc.on("close", code => {
                if (code === 0) resolve();
                else reject("Ollama pull failed");
            });

        });
    }
 
    async function downloadModel(modelName, sources){
        for(const source of sources){
            try{
                if(source === "huggingface"){
                    await downloadHF(modelName);
                    await onModelDownloaded(modelName);
                    return;
                }

                if(source === "modelscope"){
                    await downloadModelScope(modelName);
                    await onModelDownloaded(modelName);
                    return;
                }

                if(source === "verso"){
                    await downloadOllama(modelName);
                    await onModelDownloaded(modelName);
                    return;
                }
            }catch(err){
                console.warn(`Download failed from ${source}`, err);
            }
        }
        throw new Error("Model not found in selected sources");
    }

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

setInterval(draw, 50);
// Init
document.addEventListener("DOMContentLoaded", initUI);

function initAppearance(){

    lucide.createIcons();

    loadSavedAppearance();
    loadSavedTheme();
    loadSavedAccent();

    const themeButtons = document.querySelectorAll("[data-theme]");
    themeButtons.forEach(btn=>{
        btn.addEventListener("click",()=>{
            themeButtons.forEach(b=>b.classList.remove("selected"));
            btn.classList.add("selected");

            if(btn.dataset.theme === "dark"){
                document.documentElement.classList.add("dark");
                document.documentElement.classList.remove("light");
            } else {
                document.documentElement.classList.add("light");
                document.documentElement.classList.remove("dark");
            }
        });
    });

    document.querySelectorAll(
        ".local-model-toggle, .grok-model-toggle, .verso-model-toggle, .groq-model-toggle"
    ).forEach(el => {
        el.addEventListener("change", initModelSelects);
    });
}

function initBackground(){

    const bgSelect = document.getElementById("bgSelector");
    const dropZone = document.getElementById("bg-drop-zone");

    if(bgSelect){
        bgSelect.addEventListener("change", e=>{
            switchBG(e.target.value);
        });
    }

    if(dropZone){
        dropZone.addEventListener("dragover", e=>{
            e.preventDefault();
            dropZone.classList.add("border-accent");
        });

        dropZone.addEventListener("dragleave", ()=>{
            dropZone.classList.remove("border-accent");
        });

        dropZone.addEventListener("drop", e=>{
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            handleImageFile(file);
        });
    }

}

function initModelControls(){

    const modeSelector = document.getElementById("modeSelector");

    if(modeSelector){
        modeSelector.addEventListener("change", e=>{
            window.api.setMode(e.target.value);
        });
    }

    const tempRange = document.getElementById("tempRange");
    const tempValue = document.getElementById("tempValue");

    if(tempRange && tempValue){
        tempRange.addEventListener("input",()=>{
            tempValue.textContent = tempRange.value;
        });
    }

}

function initWorkflow(){

    const startBtn = document.getElementById("start-pipeline-btn");

    if(startBtn){
        startBtn.addEventListener("click", executeWorkflow);
    }

}


    function initDownloads(){

        let conf = window.api.getConfig();
        if(conf.tokenKey != null){
        const button = document.getElementById("model_confirmation");
        const input  = document.getElementById("model-search-input");
        const dropdown = document.getElementById("local-model-selector");

        if(!button) return;

        function getSelectedSources() {

        const sources = [];

        const hf = document.getElementById("hfBox")?.checked;
        const ms = document.getElementById("msBox")?.checked;

        if (hf) sources.push("huggingface");
        if (ms) sources.push("modelscope");

        return sources;
    }

    button.addEventListener("click", async ()=>{
            const modelName =
                input?.value?.length > 0
                ? input.value
                : dropdown?.value;

            if(!modelName){
                alert("Enter a model name");
                return;
            }

            const sources = getSelectedSources();

            if(sources.length === 0){
                alert("Select at least one model source");
                return;
            }

            try{

                button.disabled = true;
                button.innerText = "Downloading...";

                await downloadModel(modelName, sources);

                button.innerText = "Downloaded";

            }catch(err){
                console.error("Download failed", err);
                button.innerText = "Failed";
            }

            button.disabled = false;
        });
    }
}

function initUI() {
    initModelSelects();
    try { initAppearance(); } catch(e){ console.error("Appearance failed", e); }
    try { initBackground(); } catch(e){ console.error("Background failed", e); }
    try { initModelControls(); } catch(e){ console.error("Model controls failed", e); }
    try { initWorkflow(); } catch(e){ console.error("Workflow failed", e); }
    try { initDownloads(); } catch(e){ console.error("Downloads failed", e); }
}

window.openCreateBotModal = openCreateBotModal
window.selectBotType      = selectBotType
window.confirmCreateBot   = confirmCreateBot
window.addAgent           = addAgent;
window.setSelectedSlot    = setSelectedSlot;
window.deleteAgent        = deleteAgent;
window.createAgent        = createAgent;
window.addWorkflowStep    = addWorkflowStep;
window.executeWorkflow    = executeWorkflow;
window.closeModal         = closeModal;
window.switchTab          = switchTab;
window.deleteBot          = deleteBot;
window.handleChatSubmit   = handleChatSubmit;
window.updateWorkflowDropdowns = updateWorkflowDropdowns;