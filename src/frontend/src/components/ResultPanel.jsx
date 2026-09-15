import React from "react";
import styles from "./ResultPanel.module.css";

/**
 * ResultPanel — displays the real SignalScope prediction result or an error.
 *
 * The backend now returns a real MobileNetV2 + Grad-CAM prediction.
 * All verdicts are probabilistic: "likely AI-generated" | "likely real".
 * This component never claims certainty.
 *
 * Props
 * -----
 * appState    : "result" | "error"
 * result      : object | null
 *   { verdict, confidence, raw_prob, heatmap, explanation }
 * errorMessage: string
 */
function ResultPanel({ appState, result, errorMessage }) {
  // ------------------------------------------------------------------
  // Error state
  // ------------------------------------------------------------------
  if (appState === "error") {
    return (
      <section className={styles.panel} aria-label="Error" role="alert">
        <div className={styles.errorBadge} aria-hidden="true">⚠</div>
        <h2 className={styles.errorTitle}>Something went wrong</h2>
        <p className={styles.errorMessage}>{errorMessage}</p>
      </section>
    );
  }

  // ------------------------------------------------------------------
  // Real result from the trained model
  // ------------------------------------------------------------------
  if (result?.verdict) {
    const isAI = result.verdict === "likely AI-generated";

    return (
      <section className={styles.panel} aria-label="Analysis result">
        <h2 className={styles.resultTitle}>Analysis Result</h2>

        {/* Verdict badge */}
        <div
          className={`${styles.verdict} ${
            isAI ? styles.verdictAI : styles.verdictReal
          }`}
          role="status"
          aria-live="polite"
        >
          {result.verdict}
        </div>

        {/* Confidence bar */}
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

        {/* Raw probability — useful for understanding the model score */}
        {result.raw_prob != null && (
          <p className={styles.rawProb}>
            Model score (FAKE probability):{" "}
            <strong>{(result.raw_prob * 100).toFixed(1)}%</strong>
          </p>
        )}

        {/* Disclaimer */}
        <p className={styles.disclaimer}>
          This is a probabilistic estimate. SignalScope never claims certainty
          about whether an image is AI-generated or real.
        </p>

        {/* Grad-CAM heatmap overlay */}
        {result.heatmap && (
          <div className={styles.heatmapSection}>
            <h3 className={styles.heatmapTitle}>Grad-CAM Explanation</h3>
            <img
              src={`data:image/png;base64,${result.heatmap}`}
              alt="Grad-CAM heatmap overlay — highlighted regions influenced the model prediction"
              className={styles.heatmap}
            />
            {result.explanation && (
              <p className={styles.explanation}>{result.explanation}</p>
            )}
          </div>
        )}
      </section>
    );
  }

  // ------------------------------------------------------------------
  // Unexpected / empty response
  // ------------------------------------------------------------------
  return (
    <section className={styles.panel} aria-label="Unexpected response">
      <p className={styles.errorMessage}>
        Received an unexpected response from the server. Please try again.
      </p>
    </section>
  );
}

export default ResultPanel;
