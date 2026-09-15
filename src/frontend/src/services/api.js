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
 * API contract:
 *   POST /api/v1/predict/
 *   Body: multipart/form-data  { image: <File> }
 *
 *   Success (200):
 *   {
 *     "success":  true,
 *     "result": {
 *       "verdict":     "likely AI-generated" | "likely real",
 *       "confidence":  0.0 – 1.0,
 *       "raw_prob":    0.0 – 1.0,
 *       "heatmap":     string | null,   // base64-encoded PNG overlay
 *       "explanation": string | null    // human-readable Grad-CAM text
 *     }
 *   }
 *
 *   Error (4xx / 5xx):
 *   {
 *     "success": false,
 *     "error":   string,
 *     "message": string
 *   }
 *
 * Note: predictions are probabilistic estimates, not certainties.
 * Verdicts are always "likely AI-generated" or "likely real".
 */

const API_BASE = "/api/v1";

/**
 * Upload an image file and return SignalScope's prediction.
 *
 * @param {File} file - The image file to analyse.
 * @returns {Promise<object>} The `result` sub-object from the API response:
 *   { verdict, confidence, raw_prob, heatmap, explanation }
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
      // Do NOT set Content-Type — browser sets it with the multipart boundary.
    });
  } catch (networkError) {
    throw new Error(
      "Could not reach the SignalScope server. " +
        "Please make sure the backend is running on port 5000."
    );
  }

  // Parse JSON regardless of status code to read error details
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      `Server returned an unexpected response (HTTP ${response.status}).`
    );
  }

  if (response.ok && data.success) {
    // Return the nested result object directly so components receive:
    // { verdict, confidence, raw_prob, heatmap, explanation }
    return data.result;
  }

  // Server-side errors — surface the message to the user
  const serverMessage =
    data?.message || data?.error || `HTTP ${response.status}`;
  throw new Error(`Analysis failed: ${serverMessage}`);
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
