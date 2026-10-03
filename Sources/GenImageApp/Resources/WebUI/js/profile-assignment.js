export function profileAssignment(profile) {
  return {
    profileID: profile.id,
    profileName: profile.name,
    capability: profile.capability,
    modelID: profile.modelID,
    modelRevision: profile.modelRevision,
    architecture: profile.architecture,
  };
}

export function resolveProfileAssignment(profiles, assignment) {
  const sameCapability = profiles.filter((profile) => profile.capability === assignment.capability);
  const exact = sameCapability.find((profile) => profile.id === assignment.profileID);
  if (exact) return exact;
  // Built-in UUIDs change after relaunch. The name distinguishes base, Turbo and PE
  // profiles sharing the same checkpoint. Never guess an ambiguous legacy draft.
  const matches = sameCapability.filter((profile) =>
    profile.modelID === assignment.modelID
      && profile.modelRevision === assignment.modelRevision
      && (!assignment.architecture || profile.architecture === assignment.architecture)
      && (!assignment.profileName || profile.name === assignment.profileName),
  );
  return matches.length === 1 ? matches[0] : null;
}
