// Merge high-frequency updates without visiting assets, profiles or local drafts.
export function mergeActivityState(state, activity) {
  const models = state.models.map((model) => ({
    ...model,
    installation: activity.installations[model.descriptor.id] || model.installation,
  }));
  const structureChanged = state.isReleasingMemory !== activity.isReleasingMemory
    || state.jobs.length !== activity.jobs.length
    || state.jobs.some((job, index) => job.id !== activity.jobs[index]?.id
      || job.state !== activity.jobs[index]?.state)
    || models.some((model, index) => model.installation.phase !== state.models[index].installation.phase
      || model.installation.errorMessage !== state.models[index].installation.errorMessage);
  const nextState = {
    ...state,
    jobs: activity.jobs,
    models,
    statusMessage: activity.statusMessage ?? null,
    systemMetrics: activity.systemMetrics,
    isReleasingMemory: activity.isReleasingMemory,
  };
  return { nextState, structureChanged };
}
