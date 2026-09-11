/**
 * SignalScope API Service
 * -----------------------
 * All communication with the Flask backend goes through this module.
 * Components never call fetch() directly.
 *
 * Base URL:
 *   During development, Vite proxies /api/* → http://localhost:5000
 *   so no absolute URL is needed here.
 *
 * Future API contract (once model is connected):
 *   POST /api/v1/predict/
 *   Body: multipart/form-data  { image: <File> }
 *
 *   Success (200):
 *   {
 *     "verdict":     "likely AI-generated" | "likely real",
 *     "confidence":  0.0 – 1.0,
 *     "heatmap":     string | null,    // base64 PNG
 *     "explanation": string | null
 *   }
 *
 *   Current (501 — model not trained):
 *   {
 *     "status":   "not_implemented",
 *     "message":  string,
 *     "filename": string
 *   }
 */

const API_BASE = "/api/v1";

/**
 * Upload an image file to the SignalScope prediction endpoint.
 *
 * @param {File} file - The image file to analyse.
 * @returns {Promise<object>} Parsed JSON response from the API.
 * @throws {Error} With a user-facing message on network or server error.
 */
export async function analyseImage(file) {
  const formData = new FormData();
  formData.append("image", file);

  let response;
  try {
    response = await fetch(`${API_BASE}/predict/`, {
      method: "POST",
      body: formData,
      // Do NOT set Content-Type manually — browser sets it with the boundary.
    });
  } catch (networkError) {
    throw new Error(
      "Could not reach the SignalScope server. " +
        "Please make sure the backend is running on port 5000."
    );
  }

  // Parse JSON regardless of status code so we can read error details
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      `Server returned an unexpected response (HTTP ${response.status}).`
    );
  }

  if (response.ok || response.status === 501) {
    // Both 200 (future real result) and 501 (current dev stub) are
    // handled gracefully by the ResultPanel component.
    return data;
  }

  // 4xx errors — surface the server's error message to the user
  const serverMessage = data?.message || data?.error || "Unknown error";
  throw new Error(`Upload failed: ${serverMessage} (HTTP ${response.status})`);
}

/**
 * Check that the SignalScope API is reachable.
 *
 * @returns {Promise<object>} Health check response body.
 * @throws {Error} If the server is not reachable.
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health/`);
  if (!response.ok) {
    throw new Error(`Health check failed (HTTP ${response.status})`);
  }
  return response.json();
}
