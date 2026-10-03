import assert from "node:assert/strict";
import { test } from "node:test";
import {
  reconcileWorkspaceTabs, setActiveTabSelection, trackPendingOutput, dropPendingOutput,
} from "../../Sources/GenImageApp/Resources/WebUI/js/workspace-tabs.js";

globalThis.localStorage = { setItem() {} };
const asset = (id, kind = "generated") => ({ id, kind });
const tab = (id, assetIDs = []) => ({ id, assetIDs, selectedAssetIDs: [], selectedAssetID: null, selectionAnchorID: null });
const uiFor = (...tabs) => ({ activeWorkspaceID: "workspace", activeWorkspaceTabID: tabs[0].id,
  workspaceTabs: tabs, workspaceTabStates: {} });
const state = (assets, operations = [], jobs = [], selectedAssetID = null) => ({ assets, operations, jobs, selectedAssetID });

test("completed pending jobs route to the submitting tab, including chained outputs", () => {
  const ui = uiFor(tab("a", ["root"]), tab("b"));
  const pendingA = trackPendingOutput(ui, "generate");
  ui.activeWorkspaceTabID = "b";
  const pendingB = trackPendingOutput(ui, "generate");
  try {
    const next = state([asset("root"), asset("output"), asset("child")], [
      { action: "generate", inputAssetID: "root", outputAssetIDs: ["output"] },
      { action: "edit", inputAssetID: "output", outputAssetIDs: ["child"] },
    ], [{ id: "first", action: "generate", state: "running" },
      { id: "second", action: "generate", state: "completed" }], "child");
    reconcileWorkspaceTabs(ui, state([asset("root")]), next);
    assert.deepEqual(ui.workspaceTabs[0].assetIDs, ["root"]);
    assert.deepEqual(ui.workspaceTabs[1].assetIDs, ["output", "child"]);
    assert.equal(ui.workspaceTabs[1].selectedAssetID, "child");
    const completed = structuredClone(next);
    completed.assets.push(asset("later"));
    completed.jobs[0].state = "completed";
    completed.operations.push({ action: "generate", outputAssetIDs: ["later"] });
    reconcileWorkspaceTabs(ui, next, completed);
    assert.deepEqual(ui.workspaceTabs[0].assetIDs, ["root", "later"]);
  } finally {
    dropPendingOutput(pendingA);
    dropPendingOutput(pendingB);
  }
});

test("reconciliation preserves order and first ownership while pruning stale selections", () => {
  const a = tab("a", ["two", "one", "one", "gone"]);
  a.selectedAssetIDs = ["gone", "two", "one"];
  a.selectionAnchorID = "gone";
  const b = tab("b", ["one", "video"]);
  b.selectedAssetID = "one";
  const ui = uiFor(a, b);
  ui.activeWorkspaceTabID = "missing";
  reconcileWorkspaceTabs(ui, null, state([asset("one"), asset("two", "video"),
    asset("video", "video"), asset("unclaimed")], [], [], "one"));
  assert.equal(ui.activeWorkspaceTabID, "a");
  assert.deepEqual(a.assetIDs, ["two", "one", "one", "unclaimed"]);
  assert.deepEqual(a.selectedAssetIDs, ["one"]);
  assert.equal(a.selectionAnchorID, "one");
  assert.equal(a.selectedAssetID, "one");
  assert.deepEqual(b.assetIDs, ["video"]);
  assert.equal(b.selectedAssetID, "video");
});

test("expired and failed requests do not capture later outputs", () => {
  const ui = uiFor(tab("a", ["root"]), tab("b"));
  ui.activeWorkspaceTabID = "b";
  const expired = trackPendingOutput(ui, "generate");
  const failed = trackPendingOutput(ui, "edit");
  expired.expiresAt = 0;
  const previous = state([asset("root")], [], [{ id: "failure", action: "edit", state: "failed" }]);
  try {
    reconcileWorkspaceTabs(ui, null, previous);
    const next = state([asset("root"), asset("x"), asset("y")], [
      { action: "generate", inputAssetID: "root", outputAssetIDs: ["x"] },
      { action: "edit", inputAssetID: "x", outputAssetIDs: ["y"] },
    ], previous.jobs);
    reconcileWorkspaceTabs(ui, previous, next);
    assert.deepEqual(ui.workspaceTabs[0].assetIDs, ["root", "x", "y"]);
    assert.deepEqual(ui.workspaceTabs[1].assetIDs, []);
  } finally { dropPendingOutput(expired); dropPendingOutput(failed); }
});

test("range and additive selection preserve image order, anchors and toggle behavior", () => {
  const current = tab("a", ["one", "movie", "two", "three", "four"]);
  const ui = uiFor(current);
  const next = state([asset("one"), asset("movie", "video"), asset("two"), asset("three"), asset("four")]);
  setActiveTabSelection(ui, next, "two", {});
  setActiveTabSelection(ui, next, "four", { shiftKey: true });
  assert.deepEqual(current.selectedAssetIDs, ["two", "three", "four"]);
  setActiveTabSelection(ui, next, "one", { shiftKey: true, metaKey: true });
  assert.deepEqual(current.selectedAssetIDs, ["two", "three", "four", "one"]);
  setActiveTabSelection(ui, next, "one", { ctrlKey: true });
  assert.deepEqual(current.selectedAssetIDs, ["two", "three", "four"]);
  assert.equal(current.selectedAssetID, "four");
  setActiveTabSelection(ui, next, "movie", { shiftKey: true });
  assert.deepEqual(current.selectedAssetIDs, []);
  assert.equal(current.selectionAnchorID, null);
  assert.equal(current.selectedAssetID, "movie");
  assert.equal(setActiveTabSelection(ui, next, "missing", {}), null);
});

test("duplicate asset records keep the original any-image membership rule", () => {
  const current = tab("a", ["mixed", "end"]);
  current.selectedAssetID = "mixed";
  current.selectedAssetIDs = ["mixed"];
  current.selectionAnchorID = "mixed";
  const ui = uiFor(current);
  setActiveTabSelection(ui, state([asset("mixed", "video"), asset("mixed"), asset("end")]), "end", { shiftKey: true });
  assert.deepEqual(current.selectedAssetIDs, ["mixed", "end"]);
});
