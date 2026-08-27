import { escapeHTML } from "./format.js";
import { t } from "./i18n.js";

const automaticFlowTemplates = [
  {
    id: "simpleMV",
    version: 1,
    titleKey: "automaticFlow.simpleMV.title",
    descriptionKey: "automaticFlow.simpleMV.description",
    icon: "▶",
    requiredCapabilities: ["textToImage", "textToMusic"],
    steps: [
      { id: "visual", taskKind: "image", titleKey: "automaticFlow.simpleMV.visual" },
      { id: "music", taskKind: "music", titleKey: "automaticFlow.simpleMV.music" },
      {
        id: "imageLoop",
        taskKind: "imageLoop",
        titleKey: "automaticFlow.simpleMV.imageLoop",
        dependencyStepIDs: ["visual", "music"],
      },
      {
        id: "mediaMerge",
        taskKind: "mediaMerge",
        titleKey: "automaticFlow.simpleMV.mediaMerge",
        dependencyStepIDs: ["imageLoop", "music"],
      },
    ],
  },
];

export function renderAutomaticFlow(state) {
  return `
    <section class="page automatic-flow-page">
      <header class="page-header">
        <div class="page-header-copy">
          <h1>${t("automaticFlow.title")}</h1>
          <p>${t("automaticFlow.description")}</p>
        </div>
      </header>
      <div class="page-scroll" data-scroll-id="automatic-flow">
        <div class="automatic-flow-grid">
          ${automaticFlowTemplates.map((template) => renderFlowCard(state, template)).join("")}
        </div>
      </div>
    </section>
  `;
}

export function createAutomaticFlowPlan(state, templateID) {
  const template = automaticFlowTemplates.find((candidate) => candidate.id === templateID);
  if (!template) return null;
  const timestamp = automaticFlowTimestamp(new Date());
  const profileAssignments = template.requiredCapabilities
    .map((capability) => resolveProfileReference(state, capability))
    .filter(Boolean);
  const baseDraft = currentDraft(state, profileAssignments);
  const tabs = template.steps.map((step) => ({
    name: t(step.titleKey),
    taskKind: step.taskKind,
    promptTab: defaultPromptTab(step.taskKind),
    flow: {
      templateID: template.id,
      templateVersion: template.version,
      stepID: step.id,
      dependencyStepIDs: step.dependencyStepIDs || [],
    },
    draft: flowStepDraft(baseDraft, step),
  }));
  return {
    templateID: template.id,
    workspaceName: `${t("automaticFlow.simpleMV.workspaceName")} - ${timestamp}`,
    tabs,
  };
}

function renderFlowCard(state, template) {
  const missingCapabilities = template.requiredCapabilities.filter(
    (capability) => !resolveProfileReference(state, capability),
  );
  const busy = state.jobs.some((job) => ["queued", "running", "cancelling"].includes(job.state));
  const status = missingCapabilities.length
    ? t("automaticFlow.missingProfiles", {
        profiles: missingCapabilities.map((capability) => t(`cap.${capability}`)).join("、"),
      })
    : t("automaticFlow.ready");
  return `
    <article class="automatic-flow-card">
      <div class="automatic-flow-card-heading">
        <span class="automatic-flow-icon" aria-hidden="true">${template.icon}</span>
        <div>
          <h2>${t(template.titleKey)}</h2>
          <p>${t(template.descriptionKey)}</p>
        </div>
      </div>
      <ol class="automatic-flow-steps">
        ${template.steps.map((step) => `
          <li>
            <span>${taskKindIcon(step.taskKind)}</span>
            <strong>${t(step.titleKey)}</strong>
          </li>
        `).join("")}
      </ol>
      <div class="automatic-flow-card-footer">
        <span class="automatic-flow-status ${missingCapabilities.length ? "warning" : "ready"}">
          ${escapeHTML(status)}
        </span>
        <button
          class="primary-button"
          data-action="createAutomaticFlow"
          data-template-id="${escapeHTML(template.id)}"
          ${busy ? "disabled" : ""}
        >${t("automaticFlow.create")}</button>
      </div>
    </article>
  `;
}

function currentDraft(state, profileAssignments) {
  return {
    profileAssignments,
    recipe: { ...state.recipe, loraID: null, loraScale: 1 },
    videoOutputSettings: { ...state.videoOutputSettings },
    musicOutputSettings: { ...state.musicOutputSettings },
    subtitleSettings: { format: "srt", targetLanguage: "source" },
    imageLoopSettings: {
      width: 1280,
      height: 720,
      frameRate: 30,
      imageDurationSeconds: 5,
      totalDurationSeconds: 60,
      fitMode: "cover",
      sourceStepID: "visual",
      durationSourceStepID: "music",
    },
    mediaMergeSettings: {
      videoAssetID: "",
      audioAssetID: "",
      videoSourceStepID: "imageLoop",
      audioSourceStepID: "music",
      audioMode: "replace",
      durationMode: "shortest",
      audioVolume: 1,
    },
  };
}

function flowStepDraft(baseDraft, step) {
  const draft = JSON.parse(JSON.stringify(baseDraft));
  if (step.id === "visual") {
    draft.recipe = {
      ...draft.recipe,
      prompt: t("automaticFlow.simpleMV.imagePrompt"),
      negativePrompt: t("automaticFlow.simpleMV.imageNegativePrompt"),
      width: 1280,
      height: 720,
      steps: 9,
      outputCount: 4,
      seed: randomSeedString(),
    };
  } else if (step.id === "music") {
    draft.musicOutputSettings = {
      ...draft.musicOutputSettings,
      prompt: t("automaticFlow.simpleMV.musicPrompt"),
      lyrics: "",
      style: "anime",
      durationSeconds: 60,
      steps: 20,
      seed: randomSeedString(),
      format: "mp3",
    };
  }
  return draft;
}

function resolveProfileReference(state, capability) {
  const disabledProfileIDs = new Set(state.disabledProfileIDs || []);
  const profiles = (state.profiles || []).filter(
    (profile) => profile.capability === capability && !disabledProfileIDs.has(profile.id),
  );
  const activeID = state.activeProfileIDs?.[capability];
  const ordered = [
    ...profiles.filter((profile) => profile.id === activeID),
    ...profiles.filter((profile) => profile.id !== activeID),
  ];
  const profile = ordered.find((candidate) => profileInstalled(state, candidate));
  if (!profile) return null;
  return {
    profileID: profile.id,
    capability: profile.capability,
    modelID: profile.modelID,
    modelRevision: profile.modelRevision,
    architecture: profile.architecture,
  };
}

function profileInstalled(state, profile) {
  const requiredModelIDs = [profile.modelID, ...(profile.loras || []).map(({ modelID }) => modelID)]
    .filter((modelID, index, values) => modelID && values.indexOf(modelID) === index);
  return requiredModelIDs.length > 0 && requiredModelIDs.every((modelID) =>
    state.models.some(({ descriptor, installation }) =>
      descriptor.id === modelID && installation.phase === "installed"));
}

function defaultPromptTab(taskKind) {
  if (taskKind === "music") return "prompt";
  if (taskKind === "subtitle") return "subtitleOutput";
  if (taskKind === "imageLoop") return "imageLoopOutput";
  if (taskKind === "mediaMerge") return "mediaMergeOutput";
  return "prompt";
}

function taskKindIcon(taskKind) {
  if (taskKind === "image") return "▧";
  if (taskKind === "music") return "♫";
  if (taskKind === "imageLoop") return "▤";
  if (taskKind === "mediaMerge") return "⎌";
  return "•";
}

function automaticFlowTimestamp(date) {
  const twoDigits = (value) => String(value).padStart(2, "0");
  return `${date.getFullYear()}${twoDigits(date.getMonth() + 1)}${twoDigits(date.getDate())}-${twoDigits(date.getHours())}${twoDigits(date.getMinutes())}`;
}

function randomSeedString() {
  const upper = 2 ** 32;
  return String(Math.floor(Math.random() * upper));
}
