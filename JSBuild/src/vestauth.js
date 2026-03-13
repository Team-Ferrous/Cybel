// vestauth.js
import fetch from "node-fetch";

/*
  Quickstart
  Install vestauth and initialize your agent:

  npm i -g vestauth
  vestauth agent init

  Your agent sets secrets with a simple curl endpoint:

  vestauth agent curl -X POST https://as2.dotenvx.com/set -d '{"KEY":"value"}'
  And your agent gets secrets with a simple curl endpoint:

  vestauth agent curl "https://as2.dotenvx.com/get?key=KEY"
  That is it. This primitive unlocks secret access for agents without human-in-the-loop, oauth flows, or API keys.

  in js you can just do:   
  //Set  
  generatedtKey = SHA256.generate(...)
  await VestAuthClient.set(generatedtKey);

  //Get
  if (config.secretKey) {
      providerToken = await VestAuthClient.get(config.secretKey);
  }
*/

export class VestAuthClient {
  constructor(endpoint = "https://as2.dotenvx.com") {
    this.endpoint = endpoint;
  }

  async get(key) {
    const res = await fetch(`${this.endpoint}/get?key=${encodeURIComponent(key)}`);
    if (!res.ok) throw new Error("VestAuth fetch failed");

    const data = await res.json();
    return data.value;
  }

  async set(key, value) {
    const res = await fetch(`${this.endpoint}/set`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ [key]: value })
    });

    if (!res.ok) throw new Error("VestAuth set failed");
    return res.json();
  }
}
