import assert from "node:assert/strict";
import { test } from "node:test";
import { mergeActivityState } from "../../Sources/GenImageApp/Resources/WebUI/js/activity-state.js";

function fixture() {
  return {
    schemaVersion: 1,
    recipe: { prompt: "正在輸入的內容" },
    assets: [{ id: "image", previewURL: "genimage-asset://image" }],
    profiles: [{ id: "profile" }],
    models: [{ descriptor: { id: "model" }, installation: { phase: "downloading", progress: 0.1 } }],
    jobs: [{ id: "job", state: "running", progress: 0.2 }],
    isReleasingMemory: false,
    statusMessage: "舊通知",
  };
}
function pulse(state) {
  return {
    jobs: state.jobs.map((job) => ({ ...job, progress: 0.8 })),
    installations: { model: { phase: "downloading", progress: 0.9 } },
    systemMetrics: { ramUsage: 0.5 },
    isReleasingMemory: false,
  };
}

test("progress pulses keep local edits and media references and avoid full rendering", () => {
  const state = fixture();
  const result = mergeActivityState(state, pulse(state));
  assert.equal(result.structureChanged, false);
  assert.equal(result.nextState.recipe, state.recipe);
  assert.equal(result.nextState.assets, state.assets);
  assert.equal(result.nextState.profiles, state.profiles);
  assert.equal(result.nextState.jobs[0].progress, 0.8);
  assert.equal(result.nextState.models[0].installation.progress, 0.9);
  assert.equal(result.nextState.statusMessage, null);
  assert.equal(state.jobs[0].progress, 0.2);
});

test("completion, cancellation, model availability and memory release refresh controls", () => {
  const state = fixture();
  for (const jobState of ["completed", "cancelled", "failed"]) {
    const update = pulse(state);
    update.jobs[0].state = jobState;
    assert.equal(mergeActivityState(state, update).structureChanged, true);
  }
  const availability = pulse(state);
  availability.installations.model.phase = "installed";
  assert.equal(mergeActivityState(state, availability).structureChanged, true);
  const release = pulse(state);
  release.isReleasingMemory = true;
  assert.equal(mergeActivityState(state, release).structureChanged, true);
});
