//All available stack entries
/*const stack = {
    llm:     "mistral",
    image:   "newbie",
    model3d: "sparc3d",
    tts:     "piper",
    video:   "luma"
};*/

const activeStack = {
    llm: "mistral-7b",
    generators: [
        { type: "image", model: "newbie"  },
        { type: "3d",    model: "sparc3d" }
    ] //capped at 2 at a time in addition to the LLM, more models can be orchestrated)
      // Replacing the initial two
};

let modelRegistry = {
    llm: {
        mistral7b: {
            source: "huggingface",
            repo: "mistralai/Mistral-7B-Instruct"
        }
    },

    image: {
        newbie: {
            source: "modelscope",
            repo: "AI-ModelScope/NewBie"
        }
    },

    model3d: {
        sparc3d: {
            source: "python",
            repo: "ben-kaye/Sparc3Dsdf"
        }
    }
};

export class ModelHub {
    constructor() {
        this.hfToken    = null;                     // HuggingFace token
        this.msToken    = null;                     // ModelScope token
        this.ollamaHost = "http://localhost:11434"; // Ollama endpoint
    }

    /* ---------------- HuggingFace ---------------- */
    async loginHuggingFace() {
        window.open("https://huggingface.co/settings/tokens", "_blank");
        //const token = prompt("Enter your HuggingFace API token:");
        /*if (!token) return false;
        this.hfToken = token;
        localStorage.setItem("hfToken", token);
        alert("HuggingFace token saved!");*/
        return true;
    }

    async listHuggingFaceModels() {
        if (!this.hfToken) throw new Error("HuggingFace not authenticated");
        const res = await fetch("https://huggingface.co/api/models", {
            headers: { Authorization: `Bearer ${this.hfToken}` },
        });
        if (!res.ok) throw new Error("Failed to fetch HF models");
        return res.json();
    }

    async queryHuggingFaceModel(modelId, inputs) {
        if (!this.hfToken) throw new Error("HuggingFace not authenticated");
        const res = await fetch(`https://api-inference.huggingface.co/models/${modelId}`, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${this.hfToken}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ inputs }),
        });
        if (!res.ok) throw new Error("HF inference failed");
        return res.json();
    }

    async downloadHuggingFaceModel(modelId, file = "config.json") {
        if (!this.hfToken)
            alert("HuggingFace not authenticated")
            //throw new Error("HuggingFace not authenticated")

        const url = `https://huggingface.co/${modelId}/resolve/main/${file}`
        const res = await fetch(url, {
            headers: {
                Authorization: `Bearer ${this.hfToken}`
            }
        })

        if (!res.ok)
            throw new Error("HF download failed")

        const blob = await res.blob()
        return blob
    }


    login(){
        const token = document.getElementById("api-token-input").value; // prompt("Enter your ModelScope API token:");
        const hfCheckbox = document.getElementById("source-hf").checked
        const msCheckbox = document.getElementById("source-ms").checked
        if(msCheckbox){
            if (!token) return false;
            this.msToken = token;
            localStorage.setItem("msToken", token);
            alert("ModelScope token saved!");
        }
        if(hfCheckbox){
            this.hfToken = token;
            localStorage.setItem("hfToken", token);
            alert("HuggingFace token saved!");
            return true;
        }
    }

    /* ---------------- ModelScope ---------------- */
    async loginModelScope() {
        window.open("https://modelscope.cn/", "_blank");
        /*const token = document.getElementById("modeSelector").value; // prompt("Enter your ModelScope API token:");
        if (!token) return false;
        this.msToken = token;
        localStorage.setItem("msToken", token);
        alert("ModelScope token saved!");*/
        return true;
    }

    async listModelScopeModels() {
        if (!this.msToken) throw new Error("ModelScope not authenticated");
        const res = await fetch("https://www.modelscope.cn/api/v1/models", {
            headers: { Authorization: `Bearer ${this.msToken}` },
        });
        if (!res.ok) throw new Error("Failed to fetch ModelScope models");
        return res.json();
    }

    async queryModelScopeModel(modelId, inputs) {
        if (!this.msToken) throw new Error("ModelScope not authenticated");
        const res = await fetch(`https://www.modelscope.cn/api/v1/models/${modelId}/predict`, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${this.msToken}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ input: inputs }),
        });
        if (!res.ok) throw new Error("ModelScope inference failed");
        return res.json();
    }

    async downloadModelScopeModel(modelId) {
        if (!this.msToken)
            throw new Error("ModelScope not authenticated")
        const res = await fetch(
            `https://www.modelscope.cn/api/v1/models/${modelId}/repo`,
            {
                headers: {
                    Authorization: `Bearer ${this.msToken}`
                }
            }
        )

        if (!res.ok)
            throw new Error("ModelScope download failed")

        return res.json()
    }

    /* ---------------- Ollama ---------------- */

    setOllamaHost(host) {
        this.ollamaHost = host;
        localStorage.setItem("ollamaHost", host);
    }

    async listOllamaModels() {
        const res = await fetch(`${this.ollamaHost}/api/tags`);
        if (!res.ok) throw new Error("Failed to fetch Ollama models");
        const data = await res.json();
        return data.models || [];
    }

    async queryOllamaModel(modelId, prompt) {
        const res = await fetch(`${this.ollamaHost}/api/generate`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                model: modelId,
                prompt: prompt,
                stream: false
            }),
        });

        if (!res.ok) throw new Error("Ollama inference failed");
        return res.json();
    }

    async  downloadOllamaModel(modelId) {
        const res = await fetch(`${this.ollamaHost}/api/pull`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                name: modelId
            })
        })

        if (!res.ok) throw new Error("Ollama download failed")
        return res.json()
    }

    async downloadModel(modelId, sources) {
        if (sources.huggingface) {
            console.log("Downloading Model via HuggingFace")
            return this.downloadHuggingFaceModel(modelId)
        }
        if (sources.modelscope) {
            console.log("Downloading Model via ModelScope")
            return this.downloadModelScopeModel(modelId)
        }
        if (sources.ollama === true) {
            console.log("Downloading Model via Ollama")
            return this.downloadOllamaModel(modelId)
        }
        throw new Error("No model source selected")
    }
}

const hub = new ModelHub();

// Example usage:
async function testHF() {
    const models = await hub.listHuggingFaceModels();
    console.log("HF models:", models.slice(0,10));
}

function HF_Login() {
    window.open("https://huggingface.co/settings/tokens", "_blank");
    const token = document.getElementById("modeSelector").value; //prompt("Paste your HuggingFace token:");

    if (token) {
        localStorage.setItem("hf_token", token);
    }
}

function MS_Login() {

    window.open("https://modelscope.cn/my/access/token", "_blank");
    const token = document.getElementById("modeSelector").value; //prompt("Paste your ModelScope token:");
    if (token) {
        localStorage.setItem("ms_token", token);
    }
}

const python3DModels = {
    download_model: {
        generate: async (prompt) => {
            const res = await fetch("http://localhost:5001/download_model", {
                method: "POST",
                headers: { "Content-Type": "application/json"},
                body: JSON.stringify({ "message": prompt })
            });
            return await res.json();
        }
    },
    sparc3d: {
        generate: async (prompt) => {
            const res = await fetch("http://localhost:5001/generate_3d", {
                method: "POST",
                headers: { "Content-Type": "application/json"},
                body: JSON.stringify({ prompt })
            });

            return await res.json();
        }
    },
    newbie: {
        generate: async (prompt) => {
            const res = await fetch("http://localhost:5001/generate_image", {
                method: "POST",
                headers: { "Content-Type": "application/json"},
                body: JSON.stringify({ prompt })
            });

            return await res.json();
        }
    }
};

function findGenerator(type) {
    return activeStack.generators.find(g => g.type === type);
}

async function generate3D(prompt) {

    const res = await fetch("http://localhost:5001/generate_3d", {
        method: "POST",
        headers: { "Content-Type": "application/json"},
        body: JSON.stringify({ prompt })
    });

    return await res.json();
}

async function runPipeline(prompt) {
    const intent = detectIntent(prompt);
    if (intent === "3d") {
        return await generate3D(prompt);
    }

    if (intent === "image") {
        return await imageGen(prompt);
    }

    return await llm(prompt);
}

export {
    hub,
    activeStack,
    python3DModels,
    modelRegistry,
    findGenerator,
    runPipeline,
    HF_Login,
    MS_Login
};