export const COOKIE_NAME = "agrismart_session";
export const ONE_YEAR_MS = 365 * 24 * 60 * 60 * 1000;

// Python FastAPI owns authentication now. This helper keeps the shared
// dashboard shell compatible while routing users to the local login experience.
export const startLogin = () => {
  window.location.href = "/";
};
