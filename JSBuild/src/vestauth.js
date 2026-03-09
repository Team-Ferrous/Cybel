// vestauth.js

const fetch = require("node-fetch");

class VestAuthClient {
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

module.exports = new VestAuthClient();