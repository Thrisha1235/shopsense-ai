// Unit tests for display formatting. Run: npm test (Node's built-in test runner, no extra deps).
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  featureSetLabel,
  fmtInt,
  fmtPct,
  fmtSecondsHint,
  fmtSigned,
  formatFeatureValue,
  visitorLabel,
} from "./format.ts";

describe("number formatting", () => {
  it("formats integers and percentages", () => {
    assert.equal(fmtInt(12330), "12,330");
    assert.equal(fmtPct(0.154744525), "15.5%");
    assert.equal(fmtPct(0.154744525, 2), "15.47%");
  });

  it("signs raw-score contributions with a true minus sign", () => {
    assert.equal(fmtSigned(0.0422), "+0.042");
    assert.equal(fmtSigned(-0.0721, 4), "−0.0721");
    assert.equal(fmtSigned(0), "+0.000");
  });

  it("gives a readable duration hint", () => {
    assert.equal(fmtSecondsHint(45), "45 s");
    assert.equal(fmtSecondsHint(1200), "≈ 20.0 min");
    assert.equal(fmtSecondsHint(7200), "≈ 2.0 h");
    assert.equal(fmtSecondsHint(0), "");
  });
});

describe("feature value display", () => {
  it("formats each feature type for business users", () => {
    assert.equal(formatFeatureValue("ExitRates", 0.03), "3.00%");
    assert.equal(formatFeatureValue("ProductRelated_Duration", 1200), "1,200 s (20.0 min)");
    assert.equal(formatFeatureValue("Administrative_Duration", 45), "45 s");
    assert.equal(formatFeatureValue("Weekend", false), "No");
    assert.equal(formatFeatureValue("VisitorType", "New_Visitor"), "New visitor");
    assert.equal(formatFeatureValue("Browser", 2), "Code 2");
    assert.equal(formatFeatureValue("Month", "Nov"), "Nov");
    assert.equal(formatFeatureValue("ProductRelated", 30), "30");
    assert.equal(formatFeatureValue("PageValues", null), "—");
  });

  it("labels visitor types and feature sets", () => {
    assert.equal(visitorLabel("Returning_Visitor"), "Returning visitor");
    assert.equal(visitorLabel("Unknown"), "Unknown");
    assert.equal(featureSetLabel("with_page_values"), "With PageValues");
    assert.equal(featureSetLabel("without_page_values"), "Without PageValues");
  });
});
