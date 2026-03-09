const { HindsightClient } = require('@vectorize-io/hindsight-client');

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

// Retain a memory
await client.retain('my-bank', 'Alice works at Google');

// Recall memories
const response = await client.recall('my-bank', 'What does Alice do?');
for (const r of response.results) {
    console.log(r.text);
}

// Reflect - generate response with disposition
const answer = await client.reflect('my-bank', 'Tell me about Alice');
console.log(answer.text);

Client Initialization
import { HindsightClient } from '@vectorize-io/hindsight-client';

const client = new HindsightClient({
    baseUrl: 'http://localhost:8888',
});