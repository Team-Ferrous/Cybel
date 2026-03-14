import React, { useRef, useState } from "react";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

interface VinResult {
  vin: string;
  modelYear: number;
  mdl?: string;
  engineCylinders?: number;
  engineDisplacement?: string;
  fuelType?: string;
  status?: string;
  oToggleForExport?: boolean;
}

interface ModelRow {
  market: string;
  productLine: string;
  dorman: string;
  qtyNeeded: string;
  availableShipQTY: string;
  oe: string;
  hollander: string;
  applications: string; // 8th column
  partLocation: string;
  externalComments: string;
  newUpdated: string;
}

interface Props {
  model: {
    results: VinResult[];
    sResults: VinResult[];
    query: string;
    loadedVIN: boolean;
    loadedModel: boolean;
    loading: boolean;
    searching: boolean;
    selected?: VinResult | null;
    errorMsg?: string | null;
  };

  actions: {
    promptFileDialog: () => void;
    promptModelFileDialog: () => void;
    vinSearch: (decoded: VinResult[]) => void;
    updateQuery: (v: string) => void;
    downloadCSV: (rows: VinResult[]) => void;
    downloadPDF: (rows: VinResult[]) => void;
    toggleExport: (v: VinResult) => void;
    showDetails: (v: VinResult) => void;
    closeDetails: () => void;
    setLoading?: (v: boolean) => void; // optional action for overlay
    setLoadedVIN?: (v: boolean) => void; // optional
    setLoadedModel?: (v: boolean) => void; // optional
  };
}

function downloadCSV(rows: VinResult[]) {
  if (!rows || rows.length === 0) return;

  const headers = Object.keys(rows[0]).filter(h => h !== "status");

  const csv =
    headers.join(",") +
    "\n" +
    rows
      .map(r =>
        headers
          .map(h => `"${String((r as any)[h] ?? "").replace(/"/g, '""')}"`)
          .join(",")
      )
      .join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = "vin-results.csv";
  document.body.appendChild(link);
  link.click();

  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function downloadPDF(rows: VinResult[]) {
  const doc = new jsPDF();

  // Title
  doc.setFontSize(16);
  doc.text("VIN Decode Report", 10, 10);

  doc.setFontSize(12);
  doc.text("Decoded via CORVIN.", 10, 18);

  // Table headers
  const tableColumn = ["Model Year", "Model", "Engine Cylinders"];
  const tableRows = rows.map(r => [
    r.modelYear ?? "",
    r.mdl ?? "",
    r.engineCylinders ?? ""
  ]);

  // Generate table
  autoTable(doc, {
    head: [tableColumn],
    body: tableRows,
    startY: 25,
    theme: "grid",
    headStyles: { fillColor: [200, 200, 200] },
    styles: { cellPadding: 2, fontSize: 11 }
  });

  doc.save("vin_results.pdf");
}
// -----------------------------
// UTILS
// -----------------------------
const normalizeText = (t: string) => t.toLowerCase().trim();
const splitYearModel = (txt: string): [number, string][] => {
  const parts = txt.trim().split(" ");
  const year = parseInt(Number.isNaN(Number(parts[0])) ? "0" : parts[0]);
  const model = parts.slice(1).join(" ");
  return isNaN(year) ? [] : [[year, model]];
};

const parseYear = (val: string | number | null | undefined): number | null => {
  if (val === null || val === undefined) return null;
  if (typeof val === "string") {
    const trimmed = val.trim();
    if (trimmed === "" || trimmed.toUpperCase() === "N/A") return null;
    const n = Number(trimmed);
    return Number.isNaN(n) ? null : n;
  }
  if (typeof val === "number") return Number.isNaN(val) ? null : val;
  return null;
};

// -----------------------------
// DECODE VIN
// -----------------------------
async function decodeVin(vin: string, year?: string | number | null): Promise<VinResult> {
  const url = `http://localhost:3000/decode-vin`;

  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vin }),
    });
    const body = await resp.json();
    const data = body.Results?.[0] ?? {};

    return {
      vin,
      modelYear: parseYear(data.ModelYear) ?? 0,
      mdl: data.Model ?? null,
      engineCylinders: data.EngineCylinders ?? null,
      engineDisplacement: data.EngineDisplacement ?? null,
      fuelType: data.FuelType ?? null,
      status: "OK",
    };
  } catch (err: any) {
    return {
      vin,
      modelYear: 0,
      mdl: "NULL",
      engineCylinders: 0,
      status: err?.toString() || "Unknown error",
    };
  }
}

// -----------------------------
// BATCH VIN DECODE + FILTER
// -----------------------------
async function decodeVinCSV(vinCSV: File): Promise<VinResult[]> {
  const text = await vinCSV.text();
  const vinRows = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => {
      const [vin, yearStr] = line.split(",");
      return { vin, year: parseInt(Number.isNaN(Number(yearStr)) ? "0" : yearStr) };
    });

  const results: VinResult[] = [];
  for (const { vin, year } of vinRows) {
    const res = await decodeVin(vin, year);
    results.push(res);
  }

  return results;
}
async function parseModelCSV(modelCSV: File): Promise<Set<ModelRow>> {
  const text = await modelCSV.text();

  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return new Set();

  // Map lines to ModelRow objects
  const rows: ModelRow[] = lines.slice(1).map(line => {
    const cols = line.split(",");
    return {
      market: cols[0]?.trim() ?? "",
      productLine: cols[1]?.trim() ?? "",
      dorman: cols[2]?.trim() ?? "",
      qtyNeeded: cols[3]?.trim() ?? "",
      availableShipQTY: cols[4]?.trim() ?? "",
      oe: cols[5]?.trim() ?? "",
      hollander: cols[6]?.trim() ?? "",
      applications: cols[7]?.trim() ?? "",
      partLocation: cols[8]?.trim() ?? "",
      externalComments: cols[9]?.trim() ?? "",
      newUpdated: cols[10]?.trim() ?? ""
    };
  }).filter(r => r.applications);

  // Create a Set using a Map to deduplicate by a key (e.g., applications string)
  const uniqueSet = new Map<string, ModelRow>();
  rows.forEach(r => {
    const key = r.applications.trim().toLowerCase(); // or use year|model if you want
    if (!uniqueSet.has(key)) uniqueSet.set(key, r);
  });

  return new Set(uniqueSet.values());
}

function filterVinResults(results: VinResult[], modelRows: Set<ModelRow>): VinResult[] {
  // Build a lookup Set<string> from the ModelRows
  const keySet = new Set(
    Array.from(modelRows).flatMap(r =>
      splitYearModel(r.applications).map(([y, m]) => `${y}|${normalizeText(m)}`)
    )
  );

  return results.filter(r => {
    if (r.modelYear == null) return false;
    const model = (r.mdl ?? "").trim().toLowerCase();
    if (!model || model === "n/a") return false;
    const key = `${r.modelYear}|${model}`;
    return keySet.has(key); // ✅ now this works
  });
}

// -----------------------------
// CORVIN COMPONENT
// -----------------------------
export default function CorvinView({ model, actions }: Props) {
  const vinInputRef = useRef<HTMLInputElement>(null);
  const modelInputRef = useRef<HTMLInputElement>(null);

  const [vinCSV, setVinCSV] = useState<File | null>(null);
  const [modelCSV, setModelCSV] = useState<File | null>(null);
  const [loadedVIN, setLoadedVIN] = useState(false);
  const [loadedModel, setLoadedModel] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<VinResult[]>(model?.results ?? []);
  const [selectedEntity, setSelectedEntity] = useState<VinResult | null>(null);
  const [buyGuideRows, setBuyGuideRows] = useState<Set<ModelRow>>(new Set);

  const visibleResults = results.length > 0 ? results : results;

  const statusColors = (status?: string) => {
    if (!status || status === "Unknown") return { bg: "#FCF8E3", fg: "#8A6D3B" };
    if (status === "OK") return { bg: "#DFF0D8", fg: "#3C763D" };
    return { bg: "#F2DEDE", fg: "#A94442" };
  };

  function normalizeModelLoose(str: string) {
    return str.toLowerCase().replace(/[^a-z0-9]/g, "");
  }

    function extractModel(appModelRaw: string) {
    return normalizeModelLoose(
      appModelRaw
        .replace(/^(ford|toyota|chevy|chevrolet|dodge|gmc|nissan|honda|mazda|jeep|ram)\s+/i, "")
        .replace(/\b\d+(\.\d+)?l\b/gi, "")
        .replace(/\b(v6|v8|v10|v12)\b/gi, "")
        .replace(/\b(2wd|4wd|awd|fwd|rwd)\b/gi, "")
        .replace(/\b(with|without)\b.*$/gi, "")
        .trim()
    );
  }

  function expandModelList(raw: string): string[] {
    const cleaned = raw
      .replace(/\(.*?\)/g, "") // remove notes
      .replace(/\s+/g, " ")
      .trim();

    // split by commas or slashes
    const parts = cleaned.split(/[,\/&]/).map(p => p.trim()).filter(Boolean);

    if (parts.length === 0) return [];

    const results: string[] = [];

    // detect prefix from first entry (F250 → F)
    const prefixMatch = parts[0].match(/^([A-Za-z]+)(\d+)/);

    if (prefixMatch) {
      const prefix = prefixMatch[1];

      parts.forEach(p => {
        if (/^\d+$/.test(p)) {
          results.push(prefix + p);
        } else {
          results.push(p);
        }
      });
    } else {
      results.push(...parts);
    }

    return results.map(normalizeModelLoose);
  }

  function strictMatch(vinModel: string, model: string) {
    return vinModel === model;
  }

  function containsMatch(vinModel: string, model: string) {
    return vinModel.includes(model) || model.includes(vinModel);
  }

  function numberMatch(vinModel: string, model: string) {
    const vinDigits = vinModel.match(/\d+/);
    const modelDigits = model.match(/\d+/);

    if (!vinDigits || !modelDigits) return false;

    return vinDigits[0] === modelDigits[0];
  }

 function normalizeFordModel(str: string): string {
  let s = str.toLowerCase();

  // remove parenthetical junk
  s = s.replace(/\(.*?\)/g, "");

  // remove feature descriptions
  s = s.replace(/super\s*duty|sd|truck|pickup|series|roll\s*stability|brk|betc/g, "");

  // remove punctuation
  s = s.replace(/[^a-z0-9]/g, "");

  // normalize f-series like f250, f350
  const fMatch = s.match(/f?(\d{2,3})/); // ✅ optional f
  if (fMatch) return `f${fMatch[1]}`;

  return s;
}

function fordMatch(vinModel: string, model: string) {
  // extract numbers like 250, 350, 450
  const vinNum = vinModel.match(/\d{3,4}/);
  const modelNum = model.match(/\d{3,4}/);
  let normMod = normalizeFordModel(vinModel); 
  if (!vinNum || !modelNum || !normMod){
    console.log(model);
    return false;
  }
  // Ford trucks almost always match by number series
  return normMod;//vinNum[0] === modelNum[0];
}

function jeepMatch(vinModel: string, model: string) {
  const vinNorm = normalizeText(vinModel);
  const modelNorm = normalizeText(model);

  const jeepModels = ["wrangler", "cherokee", "grandcherokee", "compass", "patriot", "renegade"];
  const vinIsJeep = jeepModels.some(j => vinNorm.includes(j));
  const modelIsJeep = jeepModels.some(j => modelNorm.includes(j));
  if (!vinIsJeep || !modelIsJeep) return false;

  // Extract engine numbers, allow multiple engines separated by / or & 
  const extractEngines = (str: string) =>
    str.match(/\d\.\d/g) ?? []; // ["3.8","4.7"] etc.

  const vinEngines = extractEngines(vinNorm);
  const modelEngines = extractEngines(modelNorm);

  // If both have engine numbers, require at least one match
  if (vinEngines.length && modelEngines.length) {
    return vinEngines.some(ve => modelEngines.some(me => ve === me));
  }

  // Otherwise, fallback to matching just the main model name
  return jeepModels.some(j => vinNorm.includes(j) && modelNorm.includes(j));
}

function expandFordModels(appModelRaw: string): string[] {
  const parts = appModelRaw.split(",");

  if (parts.length === 1) return [appModelRaw];

  const first = parts[0].trim();

  const prefixMatch = first.match(/^[a-zA-Z]+/);
  if (!prefixMatch) return parts;

  const prefix = prefixMatch[0]; // "F"

  return parts.map(p => {
    const trimmed = p.trim();
    if (/^\d/.test(trimmed)) {
      return prefix + trimmed; // "350" -> "F350"
    }
    return trimmed;
  });
}

function vinMatchesYMM(vin: VinResult, row: ModelRow) {
  const { modelYear, mdl } = vin;
  if (!mdl || !modelYear) return false;

  // If the applications field is empty, pass through
  if (!row.applications || row.applications.trim() === "") return true;

  const vinModelNorm = normalizeModelLoose(mdl);
  const apps = row.applications.split(/[,;]/);

  return apps.some(app => {
    const match = app.match(/^(\d{4})(?:-(\d{2,4}))?\s+(.+)/);
    if (!match) return false;

    const [_, startYearStr, endYearStr, appModelRaw] = match;
    const startYear = parseInt(startYearStr, 10);
    const endYear = endYearStr ? parseInt(endYearStr, 10) : startYear;

    const modelParts = appModelRaw.split(/[\/&]/).map(p => extractModel(p));

    return (
      modelYear >= startYear &&
      modelYear <= endYear &&
      modelParts.some(
        m =>
          strictMatch(vinModelNorm, m) ||
          containsMatch(vinModelNorm, m) ||
          numberMatch(vinModelNorm, m) ||
          fordMatch(vinModelNorm, m) ||
          jeepMatch(vinModelNorm, m)
      )
    );
  });
}
  
  // -----------------------------
  // HANDLE VIN SEARCH (BATCH)
  // -----------------------------
  const handleVinSearch = async () => {
    if (!vinCSV) return alert("No VIN CSV loaded");
    setLoading(true);

    try {
      // Decode all VINs from the CSV
      const decoded = await decodeVinCSV(vinCSV);

      // Filter VINs
      const filteredResults: VinResult[] = decoded.filter(vin => {
        // PASS-THROUGH: if no model CSV is loaded, show everything
        if (!loadedModel) return true;

        // Otherwise, apply full matching logic
        return Array.from(buyGuideRows).some(row => vinMatchesYMM(vin, row));
      });

      setResults(filteredResults);
    } catch (err) {
      console.error(err);
      alert("Error decoding VINs: " + err);
    } finally {
      setLoading(false);
    }
  };

  // -----------------------------
  // UI & JSX (same as your current layout)
  // -----------------------------
  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* NAV & Inputs */}
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "1rem", padding: "1rem", borderBottom: "1px solid #ccc", flexWrap: "wrap" }}>
        <h2 style={{ marginRight: "2rem" }}>CORVIN</h2>
        <input
          ref={vinInputRef}
          type="file"
          style={{ display: "none" }}
          onChange={() => {
            const file = vinInputRef.current?.files?.[0];
            if (file) {
              setVinCSV(file);
              setLoadedVIN(true);
            }
          }}
        />
        <button style={{ minWidth: "120px" }} onClick={() => vinInputRef.current?.click()}>
          Upload VIN CSV
        </button>

        <input
          ref={modelInputRef}
          type="file"
          style={{ display: "none" }}
          onChange={async () => {
            const file = modelInputRef.current?.files?.[0];
            if (file) {
                const modelRows = await parseModelCSV(file); // ✅ now TS knows it's a File
                setModelCSV(file);
                setBuyGuideRows(modelRows);
                setLoadedModel(true);
              }
            }
          }/>
        <button style={{ minWidth: "120px" }} onClick={() => modelInputRef.current?.click()} disabled={!loadedVIN}>
          Upload Model CSV
        </button>

        <input placeholder="Enter VIN..." value={model?.query} onChange={(e) => actions.updateQuery(e.target.value)} style={{ minWidth: "200px", flex: 1 }} />

        <button style={{ minWidth: "120px" }} onClick={handleVinSearch} disabled={!loadedVIN}>
          Search
        </button>
      </div>

      {/* CSV/PDF Export Buttons */}
      {visibleResults.length > 0 && (
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "1rem", padding: "0.75rem 1rem", borderBottom: "1px solid #333", background: "#1b1b1b" }}>
          <button onClick={() => downloadCSV(visibleResults)}>Save CSV</button>
          <button onClick={() => downloadPDF(visibleResults)}>Save PDF</button>
        </div>
      )}

      {/* Overlays */}
      {loading && <Overlay text="Decoding VINs..." />}
      {!loadedVIN && !loading && <Overlay text="Ready To Load VIN CSV" />}
      {/*{loadedVIN && !loadedModel && !loading && <Overlay text="Ready To Load Model CSV" />}}

      {/* VIN Table */}
      {visibleResults.length > 0 && (
        <div style={{ flex: 1, overflowY: "auto", padding: "1rem" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", tableLayout: "fixed" }}>
            <thead>
              <tr>
                <th style={{ width: "40px" }}></th>
                <th style={{ textAlign: "left", padding: "0.75rem 1rem" }}>VIN</th>
                <th style={{ width: "120px", textAlign: "left", padding: "0.75rem 1rem" }}>Year</th>
                <th style={{ width: "180px", textAlign: "left", padding: "0.75rem 1rem" }}>Model</th>
                <th style={{ width: "120px", textAlign: "left", padding: "0.75rem 1rem" }}>Engine Cylinders</th>
                <th style={{ width: "120px", textAlign: "right", padding: "0.75rem 1rem" }}></th>
              </tr>
            </thead>
            <tbody>
              {visibleResults.map((v, i) => {
                const { bg, fg } = statusColors(v.status);
                return (
                  <tr key={`${v.vin}-${i}`} style={{ backgroundColor: bg, color: fg }}>
                    <td>
                      <input type="checkbox" checked={v.oToggleForExport ?? false} onChange={() => actions.toggleExport(v)} />
                    </td>
                    <td>{v.vin ?? "N/A"}</td>
                    <td>{v.modelYear ?? "None"}</td>
                    <td>{v.mdl ?? "N/A"}</td>
                    <td>{v.engineCylinders ?? "N/A"}</td>
                    <td>
                      <div style={{ display: "flex", justifyContent: "flex-end" }}>
                        <button onClick={() => { actions?.showDetails(v); setSelectedEntity(v); }}>Details</button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Selected VIN Details */}
      {selectedEntity && (
        <div style={{ position: "fixed", right: "20px", bottom: "20px", width: "300px", background: "rgba(50,50,50,0.9)", color: "white", border: "1px solid #ccc", borderRadius: "6px", padding: "1rem", boxShadow: "0 4px 12px rgba(0,0,0,0.2)", zIndex: 1000 }}>
          <h3>VIN Details</h3>
          <p>Year: {selectedEntity.modelYear}</p>
          <p>Model: {selectedEntity.mdl ?? "N/A"}</p>
          <p>Engine Cylinders: {selectedEntity.engineCylinders ?? "N/A"}</p>
          <p>Status: {selectedEntity.status ?? "N/A"}</p>
          <button onClick={() => setSelectedEntity(null)}>Close</button>
        </div>
      )}
    </div>
  );
}

const Overlay = ({ text }: { text: string }) => (
  <div style={{ position: "fixed", top: "20%", left: "50%", transform: "translateX(-50%)", backgroundColor: "rgba(50,50,50,0.9)", color: "white", padding: "2rem 3rem", borderRadius: "8px", zIndex: 999, fontSize: "1.2rem", textAlign: "center" }}>
    {text}
  </div>
);