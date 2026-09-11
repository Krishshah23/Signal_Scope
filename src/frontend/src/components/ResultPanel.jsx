import React from "react";
import styles from "./ResultPanel.module.css";

/**
 * ResultPanel — displays the API response or an error message.
 *
 * During Session 1, the model is not trained so the API returns
 * HTTP 501. This component handles that truthfully, showing a
 * clear development-status message rather than fake predictions.
 *
 * Props
 * -----
 * appState    : "result" | "error"
 * result      : object | null   (parsed JSON from the API)
 * errorMessage: string
 */
function ResultPanel({ appState, result, errorMessage }) {
  if (appState === "error") {
    return (
      <section className={styles.panel} aria-label="Error" role="alert">
        <div className={styles.errorBadge} aria-hidden="true">⚠</div>
        <h2 className={styles.errorTitle}>Something went wrong</h2>
        <p className={styles.errorMessage}>{errorMessage}</p>
      </section>
    );
  }

  // API returned a response — check whether it is a real result or
  // the expected development-phase 501 not-implemented response.
  const isNotImplemented = result?.status === "not_implemented";

  if (isNotImplemented) {
    return (
      <section
        className={styles.panel}
        aria-label="Analysis status"
      >
        <div className={styles.devBadge} aria-hidden="true">🔬</div>
        <h2 className={styles.devTitle}>Model Not Yet Trained</h2>
        <p className={styles.devMessage}>
          Your image was received and validated successfully. The SignalScope
          ML model has not been trained yet — real predictions will be
          available after the model training milestone is complete.
        </p>
        {result?.filename && (
          <p className={styles.devFile}>
            File received: <strong>{result.filename}</strong>
          </p>
        )}
        <div className={styles.devNote}>
          <strong>Development status:</strong> Upload pipeline ✓ &nbsp;·&nbsp;
          Model inference: pending
        </div>
      </section>
    );
  }

  // Future: real result rendering (verdict + confidence + heatmap)
  // This branch will be activated once the model is connected.
  if (result?.verdict) {
    const isAI = result.verdict === "likely AI-generated";
    return (
      <section className={styles.panel} aria-label="Analysis result">
        <h2 className={styles.resultTitle}>Analysis Result</h2>

        <div
          className={`${styles.verdict} ${isAI ? styles.verdictAI : styles.verdictReal}`}
          role="status"
          aria-live="polite"
        >
          {result.verdict}
        </div>

        {result.confidence != null && (
          <div className={styles.confidenceRow}>
            <span className={styles.confidenceLabel}>Confidence</span>
            <span className={styles.confidenceValue}>
              {(result.confidence * 100).toFixed(1)}%
            </span>
            <div
              className={styles.confidenceBar}
              role="progressbar"
              aria-valuenow={Math.round(result.confidence * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div
                className={styles.confidenceFill}
                style={{ width: `${result.confidence * 100}%` }}
              />
            </div>
          </div>
        )}

        <p className={styles.disclaimer}>
          This is a probabilistic estimate. SignalScope never claims certainty
          about whether an image is AI-generated or real.
        </p>

        {/* Heatmap placeholder — populated in Grad-CAM milestone */}
        {result.heatmap && (
          <div className={styles.heatmapSection}>
            <h3 className={styles.heatmapTitle}>Explanation Heatmap</h3>
            <img
              src={`data:image/png;base64,${result.heatmap}`}
              alt="Grad-CAM explanation heatmap"
              className={styles.heatmap}
            />
          </div>
        )}
      </section>
    );
  }

  // Unexpected API response shape
  return (
    <section className={styles.panel} aria-label="Unexpected response">
      <p className={styles.errorMessage}>
        Received an unexpected response from the server.
      </p>
    </section>
  );
}

export default ResultPanel;
