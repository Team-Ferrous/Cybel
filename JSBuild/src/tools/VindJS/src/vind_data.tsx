import Papa from "papaparse";

export interface ModelRow {
  market: string;
  productLine: string;
  dorman: string;
  qtyNeeded: string;
  availableShipQTY: string;
  oe: string;
  hollander: string;
  applications: string;
  partLocation: string;
  externalComments: string;
  newUpdated: string;
}

export interface VinResult {
  vin: string;
  modelYear: number;
  mdl?: string;
  engineCylinders?: string;
  engineDisplacement?: string;
  fuelType?: string;
  status: string;
}

export interface OutputRow {
  oYear: number;
  oModel?: string;
  oEngineCylinders?: string;
}

export function cleanVin(v: string): string {
  return v.trim().replace(/\s+/g, "").toUpperCase();
}

export function validVin(v: string): boolean {
  return cleanVin(v).length === 17;
}

export async function decodeVin(vin: string, year: number): Promise<OutputRow> {
  const clean = cleanVin(vin);

  const url = `https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/${clean}?format=json&modelyear=${year}`;

  const res = await fetch(url);
  const data = await res.json();

  const row = data?.Results?.[0];

  return {
    oYear: year,
    oModel: row?.Model?.trim() || undefined,
    oEngineCylinders: row?.EngineCylinders || undefined
  };
}

export function normalizeText(txt: string): string {
  return txt.trim().toLowerCase();
}

export function filterVinResultSet(
  keySet: Set<string>, 
  vins: VinResult[]
): VinResult[] {

  return vins.filter(vr => {
    if (!vr.mdl) return false;

    const key = `${vr.modelYear}|${normalizeText(vr.mdl)}`;
    return keySet.has(key);
  });
}

export function parseApplications(txt: string): [number, string][] {
  if (!txt) return [];

  const regex = /^(\d{4})(?:-(\d{2,4}))?\s+([^(]+)/; // "^([0-9]{4})(?:-([0-9]{2,4}))?\\s+([^(]+)"
  const match = txt.match(regex);

  if (!match) return [];

  const startYear = parseInt(match[1], 10);
  const endYearRaw = match[2];
  const model = match[3].trim();

  if (!endYearRaw) {
    return [[startYear, model]];
  }

  let endYear = parseInt(endYearRaw, 10);

  if (endYear < 100) {
    const century = Math.floor(startYear / 100) * 100;
    endYear = century + endYear;
  }

  const results: [number, string][] = [];
  for (let y = startYear; y <= endYear; y++) {
    results.push([y, model]);
  }

  return results;
}

export function parseModelCSV(file: File): Promise<ModelRow[]> {
  return new Promise((resolve, reject) => {
    Papa.parse<ModelRow>(file, {
      header: true,
      complete: results => resolve(results.data),
      error: err => reject(err)
    });
  });
}
