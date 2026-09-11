import React, { useRef } from "react";
import styles from "./UploadPanel.module.css";

/**
 * UploadPanel — image file selector, preview, and analyse button.
 *
 * Props
 * -----
 * appState    : "idle" | "loading" | "result" | "error"
 * selectedFile: File | null
 * previewUrl  : string | null   (object URL for preview)
 * onFileSelect: (File) => void
 * onAnalyse   : () => void
 * onReset     : () => void
 */
function UploadPanel({
  appState,
  selectedFile,
  previewUrl,
  onFileSelect,
  onAnalyse,
  onReset,
}) {
  const fileInputRef = useRef(null);
  const isLoading = appState === "loading";

  // ------------------------------------------------------------------
  // Drag-and-drop handlers
  // ------------------------------------------------------------------
  function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add(styles.dragOver);
  }

  function handleDragLeave(e) {
    e.currentTarget.classList.remove(styles.dragOver);
  }

  function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove(styles.dragOver);
    const file = e.dataTransfer.files?.[0];
    if (file) onFileSelect(file);
  }

  function handleInputChange(e) {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
    // Reset input so same file can be re-selected after reset
    e.target.value = "";
  }

  function handleDropZoneClick() {
    fileInputRef.current?.click();
  }

  function handleDropZoneKeyDown(e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  }

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  return (
    <section className={styles.panel} aria-label="Image upload">
      <h2 className={styles.title}>Analyse an Image</h2>
      <p className={styles.subtitle}>
        Upload an image to check whether it is likely AI-generated or likely
        real. Results are probabilistic estimates — not certainties.
      </p>

      {/* Drop zone */}
      {!previewUrl ? (
        <div
          className={styles.dropzone}
          role="button"
          tabIndex={0}
          aria-label="Click or drag an image here to upload"
          onClick={handleDropZoneClick}
          onKeyDown={handleDropZoneKeyDown}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <svg
            aria-hidden="true"
            className={styles.dropIcon}
            viewBox="0 0 48 48"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M24 4L24 32M24 4L14 14M24 4L34 14"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
            <path
              d="M8 34v6a2 2 0 002 2h28a2 2 0 002-2v-6"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinecap="round"
              fill="none"
            />
          </svg>
          <p className={styles.dropText}>
            <span className={styles.dropLink}>Click to select</span> or drag
            &amp; drop an image here
          </p>
          <p className={styles.dropHint}>JPEG, PNG, WebP, BMP — max 16 MB</p>
        </div>
      ) : (
        /* Image preview */
        <div className={styles.previewWrapper}>
          <img
            src={previewUrl}
            alt="Selected image preview"
            className={styles.preview}
          />
          <p className={styles.fileName} aria-label="Selected file">
            {selectedFile?.name}
          </p>
        </div>
      )}

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/bmp,.jpg,.jpeg,.png,.webp,.bmp"
        onChange={handleInputChange}
        className={styles.hiddenInput}
        aria-hidden="true"
        tabIndex={-1}
      />

      {/* Action buttons */}
      <div className={styles.actions}>
        {selectedFile && (
          <>
            <button
              className={styles.btnPrimary}
              onClick={onAnalyse}
              disabled={isLoading}
              aria-busy={isLoading}
            >
              {isLoading ? (
                <>
                  <span className={styles.spinner} aria-hidden="true" />
                  Analysing…
                </>
              ) : (
                "Analyse Image"
              )}
            </button>
            <button
              className={styles.btnSecondary}
              onClick={onReset}
              disabled={isLoading}
            >
              Clear
            </button>
          </>
        )}
      </div>
    </section>
  );
}

export default UploadPanel;
