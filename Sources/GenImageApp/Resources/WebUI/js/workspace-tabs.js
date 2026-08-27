// 工作區分頁的簿記：建立、還原、與 native 狀態對帳、以及每個分頁看到的資產子集。
//
// 這裡不畫任何東西，也不呼叫 render()：`ui` 與 `state` 一律由呼叫端傳入，讓分頁邏輯可以獨立
// 於 app.js 的重繪流程被閱讀與修改。

const WORKSPACE_TABS_KEY = "genimage.workspaceTabs";
const WORKSPACE_TABS_SCHEMA_VERSION = 3;
const WORKSPACE_TASK_KINDS = new Set([
  "image",
  "video",
  "music",
  "subtitle",
  "imageLoop",
  "mediaMerge",
]);

// 尚未落地的生成輸出：記住送出當下的分頁，等對應的 job 完成時把結果放回同一個分頁。
const pendingOutputs = [];

/// 送出一個會產生資產的請求前先登記；回傳的 handle 可在請求失敗時撤銷。
export function trackPendingOutput(ui, action) {
  const pending = {
    action,
    workspaceID: ui.activeWorkspaceID,
    tabID: ui.activeWorkspaceTabID,
    jobID: null,
    expiresAt: Date.now() + 60 * 60 * 1000,
  };
  pendingOutputs.push(pending);
  return pending;
}

export function dropPendingOutput(pending) {
  const index = pendingOutputs.indexOf(pending);
  if (index >= 0) pendingOutputs.splice(index, 1);
}

function bindPendingOutputJobs(previousState, nextState) {
  const previousJobIDs = new Set((previousState?.jobs || []).map((job) => job.id));
  nextState.jobs.forEach((job) => {
    if (previousJobIDs.has(job.id)) return;
    const pending = pendingOutputs.find((item) => item.action === job.action && !item.jobID);
    if (pending) pending.jobID = job.id;
  });
}

function takePendingOutputTab(action, nextState, workspaceID) {
  const now = Date.now();
  for (let index = pendingOutputs.length - 1; index >= 0; index -= 1) {
    if (pendingOutputs[index].expiresAt < now) pendingOutputs.splice(index, 1);
  }
  const completedJobIDs = new Set(
    nextState.jobs.filter((job) => job.state === "completed").map((job) => job.id),
  );
  let index = pendingOutputs.findIndex(
    (pending) => pending.workspaceID === workspaceID
      && pending.action === action
      && pending.jobID
      && completedJobIDs.has(pending.jobID),
  );
  if (index < 0) {
    index = pendingOutputs.findIndex(
      (pending) => pending.workspaceID === workspaceID && pending.action === action,
    );
  }
  if (index < 0) return null;
  return pendingOutputs.splice(index, 1)[0].tabID;
}

export function makeWorkspaceTab(options = {}) {
  const id = globalThis.crypto?.randomUUID?.() || `tab-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return {
    id,
    name: typeof options.name === "string" && options.name.trim()
      ? options.name.trim()
      : formatWorkspaceTabName(new Date()),
    taskKind: normalizeWorkspaceTaskKind(options.taskKind),
    promptTab: typeof options.promptTab === "string" ? options.promptTab : null,
    draft: cloneSerializable(options.draft),
    flow: normalizeFlowStep(options.flow),
    assetIDs: [],
    selectedAssetID: null,
    selectedAssetIDs: [],
    selectionAnchorID: null,
  };
}

export function formatWorkspaceTabName(date) {
  const twoDigits = (value) => String(value).padStart(2, "0");
  return `${twoDigits(date.getMonth() + 1)}/${twoDigits(date.getDate())} ${twoDigits(date.getHours())}:${twoDigits(date.getMinutes())}:${twoDigits(date.getSeconds())}`;
}

export function loadWorkspaceTabs() {
  try {
    const stored = JSON.parse(localStorage.getItem(WORKSPACE_TABS_KEY) || "null");
    if ([2, WORKSPACE_TABS_SCHEMA_VERSION].includes(stored?.schemaVersion) && stored.workspaces) {
      const workspaceTabStates = {};
      Object.entries(stored.workspaces).forEach(([workspaceID, value]) => {
        if (typeof workspaceID !== "string" || !workspaceID) return;
        workspaceTabStates[workspaceID] = normalizeWorkspaceTabState(value);
      });
      return { workspaceTabStates, legacyWorkspaceTabState: null };
    }
    if (stored?.tabs) {
      return {
        workspaceTabStates: {},
        legacyWorkspaceTabState: normalizeWorkspaceTabState(stored),
      };
    }
  } catch {
    localStorage.removeItem(WORKSPACE_TABS_KEY);
  }

  return {
    workspaceTabStates: {},
    legacyWorkspaceTabState: makeWorkspaceTabState(),
  };
}

export function saveWorkspaceTabs(ui) {
  persistActiveWorkspaceTabState(ui);
  localStorage.setItem(
    WORKSPACE_TABS_KEY,
    JSON.stringify({
      schemaVersion: WORKSPACE_TABS_SCHEMA_VERSION,
      workspaces: ui.workspaceTabStates,
    }),
  );
}

export function activateWorkspaceTabs(ui, workspaceID) {
  if (!workspaceID) return;
  if (ui.activeWorkspaceID === workspaceID && ui.workspaceTabs.length) return;
  persistActiveWorkspaceTabState(ui);

  let tabState = ui.workspaceTabStates[workspaceID];
  if (!tabState && ui.legacyWorkspaceTabState) {
    tabState = ui.legacyWorkspaceTabState;
    ui.legacyWorkspaceTabState = null;
  }
  if (!tabState) tabState = makeWorkspaceTabState();

  ui.workspaceTabStates[workspaceID] = tabState;
  ui.activeWorkspaceID = workspaceID;
  ui.workspaceTabs = tabState.tabs;
  ui.activeWorkspaceTabID = tabState.activeTabID;
  saveWorkspaceTabs(ui);
}

export function pruneWorkspaceTabStates(ui, workspaceIDs) {
  const validWorkspaceIDs = new Set(workspaceIDs);
  let changed = false;
  if (ui.activeWorkspaceID && !validWorkspaceIDs.has(ui.activeWorkspaceID)) {
    ui.activeWorkspaceID = null;
    ui.workspaceTabs = [];
    ui.activeWorkspaceTabID = null;
    changed = true;
  }
  Object.keys(ui.workspaceTabStates).forEach((workspaceID) => {
    if (!validWorkspaceIDs.has(workspaceID)) {
      delete ui.workspaceTabStates[workspaceID];
      changed = true;
    }
  });
  if (changed) saveWorkspaceTabs(ui);
}

function persistActiveWorkspaceTabState(ui) {
  if (!ui.activeWorkspaceID || !ui.workspaceTabs.length) return;
  ui.workspaceTabStates[ui.activeWorkspaceID] = {
    tabs: ui.workspaceTabs,
    activeTabID: ui.activeWorkspaceTabID,
  };
}

function normalizeWorkspaceTabState(value) {
  const tabs = Array.isArray(value?.tabs)
    ? value.tabs
        .filter((tab) => typeof tab?.id === "string")
        .map((tab, index) => ({
          id: tab.id,
          name: typeof tab.name === "string" && tab.name.trim()
            ? tab.name.trim()
            : formatWorkspaceTabName(new Date(Date.now() + index * 1_000)),
          taskKind: normalizeWorkspaceTaskKind(tab.taskKind || tab.generationType),
          promptTab: typeof tab.promptTab === "string" ? tab.promptTab : null,
          draft: cloneSerializable(tab.draft),
          flow: normalizeFlowStep(tab.flow),
          assetIDs: Array.isArray(tab.assetIDs)
            ? tab.assetIDs.filter((id) => typeof id === "string")
            : [],
          selectedAssetID: typeof tab.selectedAssetID === "string" ? tab.selectedAssetID : null,
          selectedAssetIDs: Array.isArray(tab.selectedAssetIDs)
            ? tab.selectedAssetIDs.filter((id) => typeof id === "string")
            : [],
          selectionAnchorID: typeof tab.selectionAnchorID === "string" ? tab.selectionAnchorID : null,
        }))
    : [];
  if (!tabs.length) return makeWorkspaceTabState();
  return {
    tabs,
    activeTabID: tabs.some((tab) => tab.id === value.activeTabID)
      ? value.activeTabID
      : tabs[0].id,
  };
}

function makeWorkspaceTabState() {
  const tab = makeWorkspaceTab();
  return { tabs: [tab], activeTabID: tab.id };
}

export function activeWorkspaceTab(ui) {
  return ui.workspaceTabs.find((tab) => tab.id === ui.activeWorkspaceTabID) || ui.workspaceTabs[0];
}

export function workspaceTabOwningAsset(ui, assetID) {
  return ui.workspaceTabs.find((tab) => tab.assetIDs.includes(assetID));
}

export function reconcileWorkspaceTabs(ui, previousState, nextState) {
  if (!ui.workspaceTabs.length) {
    const tab = makeWorkspaceTab();
    ui.workspaceTabs = [tab];
    ui.activeWorkspaceTabID = tab.id;
  }

  if (!ui.workspaceTabs.some((tab) => tab.id === ui.activeWorkspaceTabID)) {
    ui.activeWorkspaceTabID = ui.workspaceTabs[0].id;
  }

  const validAssetIDs = new Set(nextState.assets.map((asset) => asset.id));
  const imageAssetIDs = new Set(
    nextState.assets
      .filter(isImageAsset)
      .map((asset) => asset.id),
  );
  const assignedAssetIDs = new Set();
  ui.workspaceTabs.forEach((tab) => {
    tab.assetIDs = tab.assetIDs.filter((id) => validAssetIDs.has(id) && !assignedAssetIDs.has(id));
    tab.assetIDs.forEach((id) => assignedAssetIDs.add(id));
    tab.selectedAssetIDs = (tab.selectedAssetIDs || []).filter(
      (id) => tab.assetIDs.includes(id) && imageAssetIDs.has(id),
    );
    if (!tab.selectionAnchorID || !tab.selectedAssetIDs.includes(tab.selectionAnchorID)) {
      tab.selectionAnchorID = tab.selectedAssetIDs.at(-1) || null;
    }
  });

  const unassignedAssetIDs = new Set(
    nextState.assets.map((asset) => asset.id).filter((id) => !assignedAssetIDs.has(id)),
  );
  bindPendingOutputJobs(previousState, nextState);

  nextState.operations.forEach((operation) => {
    const outputIDs = operation.outputAssetIDs.filter((id) => unassignedAssetIDs.has(id));
    if (!outputIDs.length) return;

    const parentTab = operation.inputAssetID ? workspaceTabOwningAsset(ui, operation.inputAssetID) : null;
    const pendingTabID = takePendingOutputTab(
      operation.action,
      nextState,
      ui.activeWorkspaceID,
    );
    const targetTab = ui.workspaceTabs.find((tab) => tab.id === pendingTabID)
      || parentTab
      || activeWorkspaceTab(ui);
    outputIDs.forEach((id) => {
      targetTab.assetIDs.push(id);
      unassignedAssetIDs.delete(id);
    });
  });

  const fallbackTab = activeWorkspaceTab(ui);
  nextState.assets.forEach((asset) => {
    if (!unassignedAssetIDs.has(asset.id)) return;
    fallbackTab.assetIDs.push(asset.id);
    unassignedAssetIDs.delete(asset.id);
  });

  const jobStateByID = new Map(nextState.jobs.map((job) => [job.id, job.state]));
  for (let index = pendingOutputs.length - 1; index >= 0; index -= 1) {
    if (["failed", "cancelled"].includes(jobStateByID.get(pendingOutputs[index].jobID))) {
      pendingOutputs.splice(index, 1);
    }
  }

  const selectedOwner = nextState.selectedAssetID ? workspaceTabOwningAsset(ui, nextState.selectedAssetID) : null;
  if (selectedOwner) selectedOwner.selectedAssetID = nextState.selectedAssetID;
  ui.workspaceTabs.forEach((tab) => {
    if (!tab.selectedAssetID || !tab.assetIDs.includes(tab.selectedAssetID)) {
      tab.selectedAssetID = tab.assetIDs.at(-1) || null;
    }
  });
  saveWorkspaceTabs(ui);
}

export function workspaceStateForActiveTab(ui, sourceState) {
  const tab = activeWorkspaceTab(ui);
  const assetIDs = new Set(tab?.assetIDs || []);
  const assets = sourceState.assets.filter((asset) => assetIDs.has(asset.id));
  const selectedAssetID = tab?.selectedAssetID && assetIDs.has(tab.selectedAssetID)
    ? tab.selectedAssetID
    : assets.at(-1)?.id || null;
  const comparisonAssetID = sourceState.comparisonAssetID && assetIDs.has(sourceState.comparisonAssetID)
    ? sourceState.comparisonAssetID
    : null;
  const operations = sourceState.operations.filter((operation) =>
    operation.outputAssetIDs.some((id) => assetIDs.has(id)),
  );
  const selectedAssetIDs = (tab?.selectedAssetIDs || []).filter((id) => assetIDs.has(id));
  return {
    ...sourceState,
    workspaceAssets: sourceState.assets,
    workspaceOperations: sourceState.operations,
    assets,
    selectedAssetID,
    selectedAssetIDs,
    comparisonAssetID,
    operations,
  };
}

export function removeAssetFromWorkspaceTabs(ui, assetID, activeReplacementID) {
  ui.workspaceTabs.forEach((tab) => {
    tab.assetIDs = tab.assetIDs.filter((id) => id !== assetID);
    tab.selectedAssetIDs = (tab.selectedAssetIDs || []).filter((id) => id !== assetID);
    if (tab.selectionAnchorID === assetID) {
      tab.selectionAnchorID = tab.selectedAssetIDs.at(-1) || null;
    }
    if (tab.selectedAssetID !== assetID) return;
    tab.selectedAssetID = tab.id === ui.activeWorkspaceTabID
      ? activeReplacementID
      : tab.assetIDs.at(-1) || null;
  });
  saveWorkspaceTabs(ui);
}

export function setActiveTabSelection(ui, state, assetID, event) {
  const tab = activeWorkspaceTab(ui);
  if (!tab || !tab.assetIDs.includes(assetID)) return null;
  const asset = state.assets.find((item) => item.id === assetID);
  if (!asset) return null;

  const isImage = isImageAsset(asset);
  const additive = Boolean(event?.metaKey || event?.ctrlKey);
  const rangeSelection = Boolean(event?.shiftKey && isImage);
  let selectedAssetIDs = (tab.selectedAssetIDs || []).filter((id) => isImageAssetID(state, id));

  if (rangeSelection) {
    const imageIDs = tab.assetIDs.filter((id) => isImageAssetID(state, id));
    const anchorID = imageIDs.includes(tab.selectionAnchorID)
      ? tab.selectionAnchorID
      : imageIDs.includes(tab.selectedAssetID)
        ? tab.selectedAssetID
        : assetID;
    const start = imageIDs.indexOf(anchorID);
    const end = imageIDs.indexOf(assetID);
    const rangeIDs = imageIDs.slice(Math.min(start, end), Math.max(start, end) + 1);
    selectedAssetIDs = additive
      ? [...selectedAssetIDs, ...rangeIDs.filter((id) => !selectedAssetIDs.includes(id))]
      : rangeIDs;
  } else if (additive && isImage) {
    if (!selectedAssetIDs.length && isImageAssetID(state, tab.selectedAssetID)) {
      selectedAssetIDs.push(tab.selectedAssetID);
    }
    const index = selectedAssetIDs.indexOf(assetID);
    if (index >= 0 && selectedAssetIDs.length > 1) {
      selectedAssetIDs.splice(index, 1);
    } else if (index < 0) {
      selectedAssetIDs.push(assetID);
    }
    tab.selectionAnchorID = assetID;
  } else {
    selectedAssetIDs = isImage ? [assetID] : [];
    tab.selectionAnchorID = isImage ? assetID : null;
  }

  tab.selectedAssetIDs = selectedAssetIDs;
  tab.selectedAssetID = selectedAssetIDs.includes(assetID) || !isImage
    ? assetID
    : selectedAssetIDs.at(-1) || assetID;
  saveWorkspaceTabs(ui);
  return tab.selectedAssetID;
}

export function replacementAssetIDAfterRemoval(ui, state, assetID) {
  const tab = activeWorkspaceTab(ui);
  if (!tab) return null;
  if (tab.selectedAssetID !== assetID) return tab.selectedAssetID;
  const index = tab.assetIDs.indexOf(assetID);
  if (index < 0) return null;
  return tab.assetIDs[index + 1] || tab.assetIDs[index - 1] || null;
}

export function isImageAssetID(state, assetID) {
  return state.assets.some((asset) => asset.id === assetID && isImageAsset(asset));
}

export function isImageAsset(asset) {
  return ["imported", "generated", "edited", "upscaled"].includes(asset.kind);
}

function normalizeWorkspaceTaskKind(value) {
  return WORKSPACE_TASK_KINDS.has(value) ? value : "image";
}

function normalizeFlowStep(value) {
  if (!value || typeof value !== "object") return null;
  const templateID = typeof value.templateID === "string" ? value.templateID : null;
  const stepID = typeof value.stepID === "string" ? value.stepID : null;
  if (!templateID || !stepID) return null;
  return {
    templateID,
    templateVersion: Number.isInteger(value.templateVersion) ? value.templateVersion : 1,
    stepID,
    dependencyStepIDs: Array.isArray(value.dependencyStepIDs)
      ? value.dependencyStepIDs.filter((id) => typeof id === "string")
      : [],
  };
}

function cloneSerializable(value) {
  if (!value || typeof value !== "object") return null;
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return null;
  }
}
