/*const stack = {
    llm: "mistral",
    image: "newbie",
    model3d: "sparc3d",
    tts: "piper",
    video: "luma"
};*/

const activeStack = {
    llm: "mistral-7b",
    generators: [
        { type: "image", model: "newbie"  },
        { type: "3d",    model: "sparc3d" }
    ] //capped at 2 at a time in addition to the LLM, more models can be orchestrated)
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

const python3DModels = {
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

module.exports = {
    activeStack,
    python3DModels,
    modelRegistry,
    findGenerator,
    runPipeline
};