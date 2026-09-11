import React from "react";
import styles from "./Header.module.css";

/**
 * Header — top navigation bar with project name and tagline.
 */
function Header() {
  return (
    <header className={styles.header} role="banner">
      <div className={styles.inner}>
        <div className={styles.brand}>
          {/* Inline signal-wave SVG icon */}
          <svg
            aria-hidden="true"
            className={styles.icon}
            viewBox="0 0 64 32"
            xmlns="http://www.w3.org/2000/svg"
          >
            <polyline
              points="0,16 10,4 20,28 30,8 40,22 50,12 60,16"
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </svg>
          <span className={styles.name}>SignalScope</span>
        </div>
        <p className={styles.tagline}>AI-Generated Image Detection</p>
      </div>
    </header>
  );
}

export default Header;
