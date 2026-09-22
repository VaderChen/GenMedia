import assert from "node:assert/strict";
import { test } from "node:test";
import { mergeActivityState } from "../../Sources/GenImageApp/Resources/WebUI/js/activity-state.js";

// The renderer only needs these browser globals to initialize translations.
globalThis.localStorage = { getItem: () => null };
globalThis.document = { documentElement: {} };
const { renderCreationPanel } = await import("../../Sources/GenImageApp/Resources/WebUI/js/workspace.js");
const ui = { generationType: "image", promptTab: "prompt", creationCollapsed: true };

function fixture() {
  return {
    recipe: { prompt: "A red apple on a white table.", width: 256, height: 256, steps: 1 },
    assets: [], selectedAssetID: null, jobs: [],
    profiles: [{ id: "text-profile", capability: "textToImage", modelID: "qwen21" }],
    activeProfileIDs: { textToImage: "text-profile" }, disabledProfileIDs: [],
    models: [{ descriptor: { id: "qwen21" }, installation: { phase: "installed" } }],
    isReleasingMemory: false,
  };
}

function button(state, action) {
  const html = renderCreationPanel(state, ui);
  const match = html.match(new RegExp(`<button\\b[^>]*data-action="${action}"[^>]*>[\\s\\S]*?<\\/button>`));
  return match && { disabled: /\bdisabled\b/.test(match[0]), html: match[0] };
}

function complete(state, id) {
  const output = { id: `output-${id}`, kind: "generated", pixelWidth: 256, pixelHeight: 256 };
  // AppStore selects the new output before publishing the completed job.
  state.assets.push(output);
  state.selectedAssetID = output.id;
  return mergeActivityState(state, {
    jobs: state.jobs.map((job) => ({ ...job, state: "completed", progress: 1 })),
    installations: {}, systemMetrics: {}, isReleasingMemory: false,
  }).nextState;
}

test("two completed generations keep text-to-image enabled with the output selected", () => {
  let state = fixture();
  for (const id of ["first", "second"]) {
    assert.equal(button(state, "generate")?.disabled, false);
    state.jobs.push({ id, action: "generate", state: "running", progress: 0 });
    assert.equal(button(state, "generate")?.disabled, true);
    state = complete(state, id);
    assert.equal(state.selectedAssetID, `output-${id}`);
    assert.equal(button(state, "generate")?.disabled, false);
    assert.equal(button(state, "imageToImage")?.disabled, true);
  }
});

test("an available editor does not replace text-to-image and remains usable after editing", () => {
  const state = fixture();
  state.profiles.push({ id: "edit-profile", capability: "imageToImage", modelID: "qwen21" });
  state.activeProfileIDs.imageToImage = "edit-profile";
  state.assets = [{ id: "image", kind: "imported" }];
  state.selectedAssetID = "image";
  assert.equal(button(state, "generate")?.disabled, false);
  assert.equal(button(state, "imageToImage")?.disabled, false);
  state.assets.push({ id: "edit-output", kind: "edited" });
  state.selectedAssetID = "edit-output";
  state.jobs = [{ id: "edit", action: "imageToImage", state: "completed" }];
  assert.equal(button(state, "generate")?.disabled, false);
  assert.equal(button(state, "imageToImage")?.disabled, false);
  state.disabledProfileIDs = ["text-profile"];
  assert.equal(button(state, "generate")?.disabled, true);
  assert.equal(button(state, "imageToImage")?.disabled, false);
});

test("running and cancelling jobs block new generation but terminal jobs release it", () => {
  const state = fixture();
  state.assets = [{ id: "result", kind: "generated" }];
  state.selectedAssetID = "result";
  for (const status of ["queued", "running", "cancelling", "completed", "failed", "cancelled"]) {
    state.jobs = [{ id: "job", state: status }];
    assert.equal(button(state, "generate")?.disabled, ["queued", "running", "cancelling"].includes(status));
  }
  state.models[0].installation.phase = "notInstalled";
  assert.equal(button(state, "generate")?.disabled, true);
});

test("editing needs a source image and never blocks an empty text-to-image workspace", () => {
  const state = fixture();
  state.profiles.push({ id: "edit-profile", capability: "imageToImage", modelID: "qwen21" });
  state.activeProfileIDs.imageToImage = "edit-profile";
  assert.equal(button(state, "generate")?.disabled, false);
  assert.equal(button(state, "imageToImage"), null);
});
