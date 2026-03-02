const savedModels = []
const modelList = []

class LocalModel {
    constructor({ name, path, type }) {
        this.name = name
        this.path = path
        this.type = type
        this.session = null
        this.tokenizer = null
    }

    
    sample(logits, temperature = 0.7, topK = 40) {
        // apply temperature
        // filter topK
        // softmax
        // random weighted pick
    }

    registerModel(config) {
        const model = new LocalModel(config)
        savedModels.push(model)
        modelList.push({
            name: config.name,
            type: config.type
        })
    }

    
    async load() {
        this.session = await ort.InferenceSession.create(this.path, {
            executionProviders: ['webgpu']
        })
    };

    async infer(inputTokens) {
        const feeds = { input_ids: inputTokens }
        const results = await this.session.run(feeds)
        return results.logits
    };

    async generate(model, prompt, maxTokens = 100) {
        let tokens = model.tokenizer.encode(prompt)

        for (let i = 0; i < maxTokens; i++) {
            const logits = await model.infer(tokens)
            const nextToken = sample(logits)

            tokens.push(nextToken)

            const text = model.tokenizer.decode([nextToken])
            output.innerText += text

            await new Promise(r => setTimeout(r, 0)) // yield to UI
        }
    }
}
//You can then persist this:
//localStorage.setItem("models", JSON.stringify(modelList))