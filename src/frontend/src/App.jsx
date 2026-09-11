import React, { useState } from "react";
import Header from "./components/Header";
import UploadPanel from "./components/UploadPanel";
import ResultPanel from "./components/ResultPanel";
import Footer from "./components/Footer";
import { analyseImage } from "./services/api";
import styles from "./App.module.css";

/**
 * App — root component for SignalScope.
 *
 * State machine:
 *   idle     → user has not uploaded anything yet
 *   loading  → API request in flight
 *   result   → API returned a response (includes not_implemented during dev)
 *   error    → network error or unexpected failure
 */
function App() {
  const [appState, setAppState] = useState("idle"); // idle | loading | result | error
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  // ------------------------------------------------------------------
  // File selection handler
  // ------------------------------------------------------------------
  function handleFileSelect(file) {
    if (!file) return;

    // Clean up any previous object URL to avoid memory leaks
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setErrorMessage("");
    setAppState("idle");
  }

  // ------------------------------------------------------------------
  // Upload + analyse handler
  // ------------------------------------------------------------------
  async function handleAnalyse() {
    if (!selectedFile) return;

    setAppState("loading");
    setResult(null);
    setErrorMessage("");

    try {
      const data = await analyseImage(selectedFile);
      setResult(data);
      setAppState("result");
    } catch (err) {
      setErrorMessage(
        err.message || "An unexpected error occurred. Please try again."
      );
      setAppState("error");
    }
  }

  // ------------------------------------------------------------------
  // Reset handler
  // ------------------------------------------------------------------
  function handleReset() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMessage("");
    setAppState("idle");
  }

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  return (
    <div className={styles.layout}>
      <Header />

      <main className={styles.main} id="main-content">
        <UploadPanel
          appState={appState}
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          onFileSelect={handleFileSelect}
          onAnalyse={handleAnalyse}
          onReset={handleReset}
        />

        {(appState === "result" || appState === "error") && (
          <ResultPanel
            appState={appState}
            result={result}
            errorMessage={errorMessage}
          />
        )}
      </main>

      <Footer />
    </div>
  );
}

export default App;
