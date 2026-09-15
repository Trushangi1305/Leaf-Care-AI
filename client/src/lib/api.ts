export type ApiUser = {
  id: number;
  name: string;
  email: string;
  role: string;
  farmName: string;
  location: string;
  mainCrops: string;
  soilType: string;
  irrigationMethod: string;
  farmSize: string;
  memberSince: string;
  isDemo: boolean;
};

export type ScanTopK = {
  label: string;
  plant: string;
  condition: string;
  healthy: boolean;
  confidence: number;
};

export type ScanResult = {
  id: number;
  plant: string;
  condition: string;
  healthy: boolean;
  confidence: number;
  topK: ScanTopK[];
  model: string;
  createdAt: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, credentials: "include", headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(body.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

async function upload<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetch(path, { method: "POST", credentials: "include", body: formData });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(body.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

export const api = {
  auth: {
    me: () => request<{ user: ApiUser | null; authenticated: boolean }>("/api/auth/me"),
    login: (email: string, password: string) => request<{ user: ApiUser; authenticated: boolean }>("/api/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
    signup: (name: string, email: string, password: string) => request<{ user: ApiUser; authenticated: boolean }>("/api/auth/signup", { method: "POST", body: JSON.stringify({ name, email, password }) }),
    updateProfile: (profile: Omit<ApiUser, "id" | "email" | "role" | "memberSince" | "isDemo">) => request<{ user: ApiUser; updated: boolean }>("/api/auth/profile", { method: "PUT", body: JSON.stringify(profile) }),
    logout: () => request<{ success: boolean }>("/api/auth/logout", { method: "POST" }),
  },
  assistant: (message: string) => request<{ answer: string; source: string }>("/api/assistant", { method: "POST", body: JSON.stringify({ message }) }),
  farm: {
    overview: () => request<Record<string, unknown>>("/api/farm/overview"),
    scans: () => request<(Record<string, string> | ScanResult)[]>("/api/scans"),
    scan: (file: File) => {
      const formData = new FormData();
      formData.append("image", file);
      return upload<{ scan: ScanResult }>("/api/scan", formData);
    },
    recommendations: () => request<Record<string, string>[]>("/api/recommendations"),
    irrigation: () => request<Record<string, unknown>>("/api/irrigation"),
    weather: () => request<Record<string, unknown>>("/api/weather"),
  },
};
