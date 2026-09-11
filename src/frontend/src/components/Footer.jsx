import React from "react";
import styles from "./Footer.module.css";

/**
 * Footer — minimal project footer with SIH attribution.
 */
function Footer() {
  return (
    <footer className={styles.footer} role="contentinfo">
      <p className={styles.text}>
        SignalScope &mdash; SIH 2026 Internal Hackathon &nbsp;·&nbsp; Predictions are
        probabilistic estimates, not certainties.
      </p>
    </footer>
  );
}

export default Footer;
