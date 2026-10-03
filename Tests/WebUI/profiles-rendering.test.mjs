import assert from "node:assert/strict";
import { test } from "node:test";

globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.document = { documentElement: {} };
const { renderProfiles, renderProfileDetails } = await import("../../Sources/GenImageApp/Resources/WebUI/js/profiles.js");

const model = (id, phase) => ({ descriptor: { id, displayName: id }, installation: { phase } });
const profile = (id, modelID, extra = {}) => ({ id, modelID, name: id, capability: "textToImage",
  profileRevision: 1, modelRevision: "main", architecture: "mlxSwift", defaults: {}, isBuiltIn: true, ...extra });
const card = (html, id) => html.match(new RegExp(`<article[^>]*data-profile-card="${id}"[\\s\\S]*?<\\/article>`))?.[0];

test("profile ordering and actions preserve installation and active priorities", () => {
  const state = {
    models: [model("ready", "installed"), model("loading", "paused")],
    profiles: [profile("missing", "none"), profile("waiting", "loading"), profile("ready", "ready"), profile("active", "none")],
    activeProfileIDs: { textToImage: "active" }, disabledProfileIDs: ["ready"],
  };
  const html = renderProfiles(state, { profileFilter: "all" });
  const order = [...html.matchAll(/data-profile-card="([^"]+)"/g)].map((match) => match[1]);
  assert.deepEqual(order, ["active", "ready", "waiting", "missing"]);
  assert.match(card(html, "active"), /data-action="requestDeactivateProfile"/);
  assert.match(card(html, "ready"), /is-available/);
  assert.match(card(html, "waiting"), /data-action="installProfileModels"/);
  assert.doesNotMatch(card(html, "missing"), /data-action="installProfileModels"/);
});

test("duplicate model IDs retain any-installed availability and first-entry install actions", () => {
  const state = { models: [model("duplicate", "downloading"), model("duplicate", "installed")],
    profiles: [profile("p", "duplicate", { loras: [{ modelID: "duplicate" }, { modelID: "duplicate" }] })],
    activeProfileIDs: {}, disabledProfileIDs: [] };
  const html = card(renderProfiles(state, { profileFilter: "all" }), "p");
  assert.match(html, /is-available/);
  assert.match(html, /data-action="activateProfile"[^>]*>/);
  assert.doesNotMatch(html, /data-action="installProfileModels"/);
  assert.match(html, /下載中/);
});

test("activity updates and profile edits take effect on the next render", () => {
  const state = { models: [model("base", "installed"), model("lora", "paused")],
    profiles: [profile("p", "base", { isBuiltIn: false, loras: [{ modelID: "lora", scale: 0.5 }] })],
    activeProfileIDs: {}, disabledProfileIDs: [] };
  assert.match(card(renderProfiles(state, {}), "p"), /is-unavailable/);
  state.models[1].installation.phase = "installed";
  assert.match(card(renderProfiles(state, {}), "p"), /is-available/);
  state.profiles[0].loras.push({ modelID: "missing", scale: 1 });
  assert.match(card(renderProfiles(state, {}), "p"), /is-unavailable/);
  const details = renderProfileDetails(state.profiles[0], false, state.models);
  assert.match(details, /lora · 50%/);
  assert.match(details, /data-action="saveProfile"/);
  assert.match(details, /data-action="activateProfile"[^>]*disabled/);
});
