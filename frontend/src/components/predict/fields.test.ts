// Unit tests for the prediction form: field coverage, validation and API payload conversion.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ALL_FIELDS, EMPTY, EXAMPLE, QUICK_BOUNCE, toSession, validateField, type Field } from "./fields.ts";

// The 16 inputs required by the served model (backend train.required_features without PageValues).
const MODEL_FEATURES = [
  "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
  "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "SpecialDay",
  "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend",
];

const field = (key: string): Field => {
  const f = ALL_FIELDS.find((x) => x.key === key);
  assert.ok(f, `missing field ${key}`);
  return f;
};

describe("form fields", () => {
  it("cover exactly the 16 model inputs (no PageValues)", () => {
    assert.deepEqual(ALL_FIELDS.map((f) => f.key).sort(), [...MODEL_FEATURES].sort());
  });

  it("every field has a label and help text", () => {
    for (const f of ALL_FIELDS) {
      assert.ok(f.label.length > 2, f.key);
      assert.ok(f.help.length > 5, f.key);
    }
  });
});

describe("validation mirrors the API limits", () => {
  it("rejects empty, non-numeric, fractional counts and out-of-range values", () => {
    assert.equal(validateField(field("ProductRelated"), ""), "Required.");
    assert.equal(validateField(field("ProductRelated"), "abc"), "Enter a number.");
    assert.equal(validateField(field("ProductRelated"), "2.5"), "Enter a whole number.");
    assert.equal(validateField(field("ProductRelated"), "-1"), "Must be at least 0.");
    assert.equal(validateField(field("ProductRelated"), "10001"), "Must be at most 10,000.");
    assert.equal(validateField(field("ProductRelated_Duration"), "86401"), "Must be at most 86,400.");
    assert.equal(validateField(field("BounceRates"), "101"), "Must be at most 100%.");
  });

  it("accepts valid values and toggles", () => {
    assert.equal(validateField(field("ProductRelated"), "30"), null);
    assert.equal(validateField(field("ExitRates"), "3.5"), null);
    assert.equal(validateField(field("Weekend"), ""), null);
    assert.equal(validateField(field("Month"), "Nov"), null);
  });

  it("the samples are valid and Clear empties every field", () => {
    for (const preset of [EXAMPLE, QUICK_BOUNCE]) {
      for (const f of ALL_FIELDS) assert.equal(validateField(f, preset[f.key]), null, `${f.key}=${preset[f.key]}`);
    }
    assert.ok(Object.values(EMPTY).every((v) => v === ""));
  });
});

describe("toSession builds the exact API payload", () => {
  it("converts percentages to fractions and types every field", () => {
    const s = toSession(EXAMPLE);
    assert.deepEqual(Object.keys(s).sort(), [...MODEL_FEATURES].sort());
    assert.equal(s.BounceRates, 0.01);
    assert.equal(s.ExitRates, 0.03);
    assert.equal(s.ProductRelated_Duration, 1200);
    assert.equal(s.Weekend, false);
    assert.equal(s.Month, "Nov");
    assert.equal(s.VisitorType, "Returning_Visitor");
    assert.equal(typeof s.Browser, "number");
  });

  it("matches the documented API example session", () => {
    // backend/app/schemas.py EXAMPLE_SESSION
    assert.deepEqual(toSession(EXAMPLE), {
      Administrative: 2, Administrative_Duration: 45, Informational: 0, Informational_Duration: 0,
      ProductRelated: 30, ProductRelated_Duration: 1200, BounceRates: 0.01, ExitRates: 0.03, SpecialDay: 0,
      Month: "Nov", OperatingSystems: 2, Browser: 2, Region: 1, TrafficType: 2,
      VisitorType: "Returning_Visitor", Weekend: false,
    });
  });
});
