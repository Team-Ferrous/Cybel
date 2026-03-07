// hf-modelscope.js
export class ModelHub {
    constructor() {
        this.hfToken = null;          // HuggingFace token
        this.msToken = null;          // ModelScope token
    }

    /* ---------------- HuggingFace ---------------- */
    async loginHuggingFace() {
        // Prompt for token manually (or use popup flow if available)
        const token = prompt("Enter your HuggingFace API token:");
        if (!token) return false;
        this.hfToken = token;
        localStorage.setItem("hfToken", token);
        alert("HuggingFace token saved!");
        return true;
    }

    async listHuggingFaceModels() {
        if (!this.hfToken) throw new Error("HuggingFace not authenticated");
        const res = await fetch("https://huggingface.co/api/models", {
            headers: { Authorization: `Bearer ${this.hfToken}` },
        });
        if (!res.ok) throw new Error("Failed to fetch HF models");
        return res.json(); // Array of models
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

    /* ---------------- ModelScope ---------------- */
    async loginModelScope() {
        // ModelScope also uses a token
        const token = prompt("Enter your ModelScope API token:");
        if (!token) return false;
        this.msToken = token;
        localStorage.setItem("msToken", token);
        alert("ModelScope token saved!");
        return true;
    }

    async listModelScopeModels() {
        if (!this.msToken) throw new Error("ModelScope not authenticated");
        const res = await fetch("https://www.modelscope.cn/api/v1/models", {
            headers: { Authorization: `Bearer ${this.msToken}` },
        });
        if (!res.ok) throw new Error("Failed to fetch ModelScope models");
        return res.json(); // Array of models
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
}