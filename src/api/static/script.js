/**
 * Project-RAG_LLM Enterprise Studio — Stateful Frontend Controller
 * Layer: L5A (Stateful API-Connected Views)
 * Contract: ZERO emojis, 100% inline SVG icons, strict JWT RBAC & multi-tenant state isolation.
 */

// ---------------------------------------------------------------------------
// 1. STATE STORE & PRE-SEEDED DEMO PERSONAS
// ---------------------------------------------------------------------------
const state = {
  currentUser: null,
  token: null,
  mode: 'hybrid_rerank',
  provider: 'lmstudio',
  geminiApiKey: localStorage.getItem('rag_gemini_api_key') || '',
  activeSessionId: localStorage.getItem('rag_active_session_id') || null,
  sessionTurnCount: 0,
  pollingIntervals: {},
  pollingTimeouts: {},
  uploadQueue: {},
  cooldownCountdownTimer: null,
};

const DEMO_PERSONAS = {
  admin_default: {
    email: 'admin@default.org',
    password: 'adminpassword123',
    label: 'Admin (Default)',
  },
  manager_default: {
    email: 'manager@default.org',
    password: 'managerpassword123',
    label: 'Manager (Default)',
  },
  analyst_default: {
    email: 'analyst@default.org',
    password: 'analystpassword123',
    label: 'Analyst (Default)',
  },
  viewer_default: {
    email: 'viewer@default.org',
    password: 'viewerpassword123',
    label: 'Viewer (Default)',
  },
  admin_tenant_b: {
    email: 'admin@tenant_b.org',
    password: 'tenantbadmin123',
    label: 'Admin (Tenant B)',
  },
  viewer_tenant_b: {
    email: 'viewer@tenant_b.org',
    password: 'tenantbviewer123',
    label: 'Viewer (Tenant B)',
  },
};

// ---------------------------------------------------------------------------
// 2. INLINE SVG ICON FACTORIES (ZERO EMOJIS)
// ---------------------------------------------------------------------------
const SVG_ICONS = {
  user: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>',
  trash: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>',
  alert: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
  shield: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
  clock: '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>',
  cpu: '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect></svg>',
  dollar: '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>',
  layers: '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
  file: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>',
};

// ---------------------------------------------------------------------------
// 3. UTILITY HELPERS
// ---------------------------------------------------------------------------
function escapeHtml(raw) {
  if (raw === null || raw === undefined) return '';
  return String(raw)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function getAuthHeaders(extraHeaders = {}) {
  const headers = { ...extraHeaders };
  if (state.token) {
    headers['Authorization'] = `Bearer ${state.token}`;
  }
  if (state.geminiApiKey) {
    headers['X-Gemini-API-Key'] = state.geminiApiKey;
  }
  return headers;
}

async function safeParseJson(res) {
  try {
    return await res.json();
  } catch (_err) {
    return {};
  }
}

function formatHttpError(res, data, fallbackMsg) {
  if (res.status === 429) {
    const retryAfter = res.headers.get('Retry-After') || '60';
    return `Rate limit exceeded (HTTP 429). Retry after ${retryAfter}s.`;
  }
  if (res.status === 401) {
    return data.detail || 'Authentication session expired (HTTP 401). Please switch or refresh persona.';
  }
  return data.detail || `${fallbackMsg} (HTTP ${res.status})`;
}

function roleClassFor(role) {
  const normalized = (role || 'VIEWER').toUpperCase();
  if (normalized === 'ADMIN') return 'role-admin';
  if (normalized === 'MANAGER') return 'role-manager';
  if (normalized === 'ANALYST') return 'role-analyst';
  return 'role-viewer';
}

function clearAllPollingIntervals() {
  Object.keys(state.pollingIntervals).forEach((jobId) => {
    clearInterval(state.pollingIntervals[jobId]);
    clearTimeout(state.pollingIntervals[jobId]);
    delete state.pollingIntervals[jobId];
  });
  if (state.pollingTimeouts) {
    Object.keys(state.pollingTimeouts).forEach((jobId) => {
      clearTimeout(state.pollingTimeouts[jobId]);
      delete state.pollingTimeouts[jobId];
    });
  }
  state.uploadQueue = {};
  hide429CooldownBanner();
}

function setConnectionStatus(online, labelText = null) {
  const statusDot = document.getElementById('connection-status');
  const statusLabel = document.querySelector('.status-label');
  if (statusDot) {
    statusDot.classList.toggle('status-online', online);
    statusDot.classList.toggle('status-offline', !online);
  }
  if (statusLabel) {
    statusLabel.textContent = labelText || (online ? 'Connected' : 'Offline');
  }
}

// ---------------------------------------------------------------------------
// 4. MULTI-TENANT JWT STATE MACHINE & PERSONA SWITCHER
// ---------------------------------------------------------------------------
function getInitials(nameOrEmail) {
  if (!nameOrEmail) return 'AD';
  const clean = nameOrEmail.split('@')[0].replace(/[^a-zA-Z0-9]/g, ' ').trim();
  const parts = clean.split(/\s+/).filter(Boolean);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[1][0]).toUpperCase();
  }
  return (clean.slice(0, 2) || 'AD').toUpperCase();
}

function updateActiveUserBadge(user) {
  const badge = document.getElementById('active-user-badge');
  const avatarInitials = document.getElementById('avatar-initials');
  if (!user) return;
  const roleUpper = (user.role || 'VIEWER').toUpperCase();
  const roleClass = roleClassFor(roleUpper);
  const initials = getInitials(user.email || user.role);

  if (avatarInitials) {
    avatarInitials.textContent = initials;
  }
  if (badge) {
    badge.innerHTML = `
      ${SVG_ICONS.user}
      <span class="badge-email" id="current-user-email">${escapeHtml(user.email)}</span>
      <span class="badge-org" id="current-user-org">${escapeHtml(user.org_id)}</span>
      <span class="badge-role ${roleClass}" id="current-user-role">${escapeHtml(roleUpper)}</span>
    `;
  }
  updateProfileDrawer(user);
}

function updateProfileDrawer(user) {
  if (!user) return;
  const drawerAvatarInitials = document.getElementById('drawer-avatar-initials');
  const drawerEmail = document.getElementById('drawer-user-email');
  const drawerOrg = document.getElementById('drawer-user-org');
  const drawerRole = document.getElementById('drawer-user-role');

  const roleUpper = (user.role || 'VIEWER').toUpperCase();
  const roleClass = roleClassFor(roleUpper);
  const initials = getInitials(user.email || user.role);

  if (drawerAvatarInitials) drawerAvatarInitials.textContent = initials;
  if (drawerEmail) drawerEmail.textContent = user.email;
  if (drawerOrg) drawerOrg.textContent = `[${user.org_id || 'default_org'}]`;
  if (drawerRole) {
    drawerRole.textContent = roleUpper;
    drawerRole.className = `badge-role ${roleClass}`;
  }

  // Clearances Matrix
  const publicStatus = document.getElementById('clearance-public-status');
  const orgStatus = document.getElementById('clearance-org-status');
  const restrictedStatus = document.getElementById('clearance-restricted-status');

  if (publicStatus) {
    publicStatus.textContent = 'Granted';
    publicStatus.className = 'clearance-status granted';
  }
  if (orgStatus) {
    orgStatus.textContent = 'Granted';
    orgStatus.className = 'clearance-status granted';
  }
  if (restrictedStatus) {
    const hasRestricted = ['ADMIN', 'MANAGER'].includes(roleUpper);
    restrictedStatus.textContent = hasRestricted ? 'Granted' : 'Denied';
    restrictedStatus.className = `clearance-status ${hasRestricted ? 'granted' : 'denied'}`;
  }

  // Capabilities chips
  const chipsContainer = document.getElementById('role-capabilities-chips');
  if (chipsContainer) {
    let chips = [];
    if (roleUpper === 'ADMIN') {
      chips = ['Full RBAC', 'Document Ingestion', 'Metric Reset', 'Cross-Tenant Switch', 'Grounded Retrieval'];
    } else if (roleUpper === 'MANAGER') {
      chips = ['Document Ingestion', 'Restricted Docs', 'Team View', 'Grounded Retrieval'];
    } else if (roleUpper === 'ANALYST') {
      chips = ['Org Scope Docs', 'Public Docs', 'Telemetry Telemetry', 'Grounded Retrieval'];
    } else {
      chips = ['Public Scope Docs', 'Grounded Retrieval'];
    }
    chipsContainer.innerHTML = chips
      .map((chip) => `<span class="capability-chip active">${escapeHtml(chip)}</span>`)
      .join('');
  }

  // Team management section visibility
  const teamSection = document.getElementById('team-management-section');
  if (teamSection) {
    const canManageTeam = ['ADMIN', 'MANAGER'].includes(roleUpper);
    teamSection.classList.toggle('hidden', !canManageTeam);
    if (canManageTeam) {
      loadTeamMembers(roleUpper === 'ADMIN');
    }
  }
}

async function loadTeamMembers(isAdmin) {
  const teamList = document.getElementById('team-users-list');
  if (!teamList) return;

  try {
    const res = await fetch('/api/auth/users', {
      headers: getAuthHeaders(),
    });
    if (!res.ok) {
      teamList.innerHTML = `<p class="text-dim text-sm">Unable to load team members (${res.status}).</p>`;
      return;
    }
    const users = await safeParseJson(res);
    if (!Array.isArray(users) || users.length === 0) {
      teamList.innerHTML = `<p class="text-dim text-sm">No organization users found.</p>`;
      return;
    }

    teamList.innerHTML = users
      .map((u) => {
        const uRole = (u.role || 'VIEWER').toUpperCase();
        const uRoleClass = roleClassFor(uRole);
        const isSelf = state.currentUser && state.currentUser.id === u.id;
        const controls = isAdmin
          ? `<select class="team-role-select" data-user-id="${escapeHtml(u.id)}" ${isSelf ? 'disabled' : ''}>
              <option value="ADMIN" ${uRole === 'ADMIN' ? 'selected' : ''}>ADMIN</option>
              <option value="MANAGER" ${uRole === 'MANAGER' ? 'selected' : ''}>MANAGER</option>
              <option value="ANALYST" ${uRole === 'ANALYST' ? 'selected' : ''}>ANALYST</option>
              <option value="VIEWER" ${uRole === 'VIEWER' ? 'selected' : ''}>VIEWER</option>
            </select>`
          : `<span class="badge-role ${uRoleClass}">${escapeHtml(uRole)}</span>`;

        return `
          <div class="team-user-row" data-user-row="${escapeHtml(u.id)}">
            <div class="team-user-info">
              <span class="team-user-email" title="${escapeHtml(u.email)}">${escapeHtml(u.email)} ${isSelf ? '(you)' : ''}</span>
              <span class="team-user-meta">${escapeHtml(u.org_id || 'default_org')}</span>
            </div>
            <div class="team-user-controls">
              ${controls}
            </div>
          </div>
        `;
      })
      .join('');

    // Attach change handlers if Admin
    if (isAdmin) {
      teamList.querySelectorAll('.team-role-select').forEach((sel) => {
        sel.addEventListener('change', async (e) => {
          const userId = e.target.dataset.userId;
          const newRole = e.target.value;
          try {
            e.target.disabled = true;
            const patchRes = await fetch(`/api/auth/users/${encodeURIComponent(userId)}`, {
              method: 'PATCH',
              headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
              body: JSON.stringify({ role: newRole }),
            });
            if (patchRes.ok) {
              await loadTeamMembers(true);
            } else {
              const errData = await safeParseJson(patchRes);
              alert(formatHttpError(patchRes, errData, 'Failed to update user role'));
              await loadTeamMembers(true);
            }
          } catch (err) {
            console.error('Role update error:', err);
            await loadTeamMembers(true);
          } finally {
            e.target.disabled = false;
          }
        });
      });
    }
  } catch (err) {
    console.error('Failed to load team members:', err);
    teamList.innerHTML = `<p class="text-dim text-sm">Failed to load team members.</p>`;
  }
}

async function loadStorageStatus() {
  const localBtn = document.getElementById('storage-mode-local-btn');
  const cloudBtn = document.getElementById('storage-mode-cloud-btn');
  const cloudFields = document.getElementById('cloud-storage-fields');
  const cloudQdrantUrl = document.getElementById('cloud-qdrant-url');
  const cloudDbUrl = document.getElementById('cloud-database-url');

  try {
    const res = await fetch('/api/config/storage-status', {
      headers: getAuthHeaders(),
    });
    if (res.ok) {
      const data = await safeParseJson(res);
      const isCloud = data.storage_mode === 'cloud' || Boolean(data.is_cloud);
      state.storageMode = isCloud ? 'cloud' : 'local';

      if (localBtn && cloudBtn) {
        localBtn.classList.toggle('active', !isCloud);
        cloudBtn.classList.toggle('active', isCloud);
      }
      if (cloudFields) {
        cloudFields.classList.toggle('hidden', !isCloud);
      }
      if (cloudQdrantUrl && data.cloud_qdrant_url) {
        cloudQdrantUrl.value = data.cloud_qdrant_url;
      }
      if (cloudDbUrl && data.cloud_database_url) {
        cloudDbUrl.value = data.cloud_database_url;
      }
    }
  } catch (err) {
    console.error('Failed to load storage status:', err);
  }
}

function applyRolePermissions(user) {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const docVisibility = document.getElementById('doc-visibility');
  const docRolesFieldset = document.getElementById('doc-roles');
  const resetMetricsBtn = document.getElementById('reset-metrics-btn');
  const dropzoneText = dropZone ? dropZone.querySelector('.dropzone-text') : null;

  const roleUpper = user && user.role ? user.role.toUpperCase() : '';
  const canUpload = Boolean(['ADMIN', 'MANAGER'].includes(roleUpper));
  const isAdmin = roleUpper === 'ADMIN';

  if (dropZone) {
    dropZone.classList.toggle('disabled', !canUpload);
    dropZone.style.opacity = canUpload ? '1' : '0.45';
    dropZone.style.pointerEvents = canUpload ? 'auto' : 'none';
    dropZone.setAttribute('aria-disabled', canUpload ? 'false' : 'true');
  }
  if (dropzoneText) {
    dropzoneText.textContent = canUpload
      ? 'Drag and drop PDF here'
      : 'Upload restricted (ADMIN / MANAGER only)';
  }
  if (fileInput) {
    fileInput.disabled = !canUpload;
  }
  if (docVisibility) {
    docVisibility.disabled = !canUpload;
  }
  if (docRolesFieldset) {
    const checkboxes = docRolesFieldset.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach((cb) => {
      if (cb.value === 'ADMIN') {
        cb.disabled = true;
      } else {
        cb.disabled = !canUpload;
      }
    });
  }
  if (resetMetricsBtn) {
    resetMetricsBtn.classList.toggle('hidden', !isAdmin);
  }
}

function updateGeminiKeyStatusUI() {
  const statusEl = document.getElementById('gemini-key-status');
  const inputEl = document.getElementById('gemini-api-key');
  if (!statusEl) return;
  const hasKey = Boolean(state.geminiApiKey && state.geminiApiKey.trim());
  statusEl.textContent = hasKey ? 'Active' : 'No Key Set';
  statusEl.classList.toggle('chip-active', hasKey);
  statusEl.classList.toggle('chip-inactive', !hasKey);
  if (inputEl && hasKey && !inputEl.value) {
    inputEl.value = state.geminiApiKey;
  }
}

async function checkLlmStatus() {
  try {
    const res = await fetch('/api/config/llm-status', {
      headers: getAuthHeaders(),
    });
    if (res.ok) {
      const data = await safeParseJson(res);
      const geminiConfigured = Boolean(
        data.providers && data.providers.gemini && (data.providers.gemini.configured || data.providers.gemini.available)
      );
      const statusEl = document.getElementById('gemini-key-status');
      if (statusEl && geminiConfigured && !state.geminiApiKey) {
        statusEl.textContent = 'Server Config';
        statusEl.classList.add('chip-active');
        statusEl.classList.remove('chip-inactive');
      }
    }
  } catch (err) {
    console.error('LLM status check error:', err);
  }
}

async function loginPersona(personaKey) {
  const credentials = DEMO_PERSONAS[personaKey];
  if (!credentials) return;

  clearAllPollingIntervals();

  // Highlight selected persona button immediately for responsive feedback
  const personaBtns = document.querySelectorAll('.persona-btn[data-persona]');
  personaBtns.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.persona === personaKey);
  });

  try {
    let res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: credentials.email,
        password: credentials.password,
      }),
    });

    // Auto-seed demo accounts on fresh database if 401 returned
    if (res.status === 401) {
      await fetch('/api/auth/seed', { method: 'POST' });
      res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: credentials.email,
          password: credentials.password,
        }),
      });
    }

    if (!res.ok) {
      setConnectionStatus(false, 'Auth Error');
      return;
    }

    const data = await res.json();
    state.token = data.access_token;
    state.currentUser = data.user;

    const rememberCb = document.getElementById('login-remember-me');
    if (!rememberCb || rememberCb.checked || localStorage.getItem('rag_access_token')) {
      localStorage.setItem('rag_access_token', data.access_token);
    }

    const loginGate = document.getElementById('login-gate-overlay');
    if (loginGate) loginGate.classList.add('hidden');

    setConnectionStatus(true, 'Connected');
    updateActiveUserBadge(state.currentUser);
    applyRolePermissions(state.currentUser);

    // Visually indicate tenant context switch in workspace
    const chatHistory = document.getElementById('chat-history');
    if (chatHistory) {
      chatHistory.innerHTML = `
        <div class="empty-state">
          <div class="icon-pulse">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
          </div>
          <h2>Ready to analyze</h2>
          <p>Switched to <strong>${escapeHtml(credentials.label)}</strong> (Tenant: <code>${escapeHtml(state.currentUser.org_id)}</code>, Tier: <code>${escapeHtml(state.currentUser.role)}</code>).</p>
        </div>
      `;
    }

    await loadDocumentCatalog();

    if (typeof ChatSessionController !== 'undefined' && ChatSessionController.syncSession) {
      await ChatSessionController.syncSession();
    }

    const drawer = document.getElementById('metrics-drawer');
    if (drawer && !drawer.classList.contains('hidden')) {
      await loadMetrics();
    }
  } catch (err) {
    console.error('Persona login error:', err);
    setConnectionStatus(false, 'Offline');
  }
}

// ---------------------------------------------------------------------------
// 5. ASYNC 202 INGESTION POLLING & DOCUMENT CATALOG MANAGER
// ---------------------------------------------------------------------------
function updateUploadProgressUI(percent, message, isError = false, isSuccess = false) {
  const statusCard = document.getElementById('upload-status-card');
  const progressBar = document.getElementById('upload-progress-bar');
  const progressPercent = document.getElementById('upload-progress-percent');
  const statusText = document.getElementById('upload-status-text');

  if (statusCard) {
    statusCard.classList.remove('hidden');
  }
  const clamped = Math.max(0, Math.min(100, Math.round(percent)));
  if (progressBar) {
    progressBar.style.width = `${clamped}%`;
  }
  if (progressPercent) {
    progressPercent.textContent = `${clamped}%`;
  }
  if (statusText) {
    statusText.textContent = message;
    statusText.classList.toggle('text-error', Boolean(isError));
    statusText.classList.toggle('text-success', Boolean(isSuccess));
  }
}

function show429CooldownBanner(seconds) {
  const banner = document.getElementById('upload-cooldown-banner');
  const textEl = document.getElementById('upload-cooldown-text');
  const statusCard = document.getElementById('upload-status-card');
  if (!banner || !textEl) return;
  if (statusCard) statusCard.classList.remove('hidden');
  banner.classList.remove('hidden');

  if (state.cooldownCountdownTimer) {
    clearInterval(state.cooldownCountdownTimer);
  }

  let remaining = Math.max(1, Math.round(seconds));
  textEl.textContent = `Rate limit cooldown · resuming in ${remaining}s...`;

  state.cooldownCountdownTimer = setInterval(() => {
    remaining -= 1;
    if (remaining <= 0) {
      clearInterval(state.cooldownCountdownTimer);
      state.cooldownCountdownTimer = null;
      banner.classList.add('hidden');
    } else {
      textEl.textContent = `Rate limit cooldown · resuming in ${remaining}s...`;
    }
  }, 1000);
}

function hide429CooldownBanner() {
  if (state.cooldownCountdownTimer) {
    clearInterval(state.cooldownCountdownTimer);
    state.cooldownCountdownTimer = null;
  }
  const banner = document.getElementById('upload-cooldown-banner');
  if (banner) banner.classList.add('hidden');
}

function renderUploadQueueUI() {
  const statusCard = document.getElementById('upload-status-card');
  const queueList = document.getElementById('upload-queue-list');
  const progressBar = document.getElementById('upload-progress-bar');
  const progressPercent = document.getElementById('upload-progress-percent');
  const statusText = document.getElementById('upload-status-text');

  const jobs = Object.values(state.uploadQueue);
  if (jobs.length === 0) return;

  if (statusCard) statusCard.classList.remove('hidden');

  // Identify active (PROCESSING) job or highest priority PENDING job
  const activeJob =
    jobs.find((j) => (j.status || '').toUpperCase() === 'PROCESSING') ||
    jobs.find((j) => (j.status || '').toUpperCase() === 'PENDING') ||
    jobs[jobs.length - 1];

  if (activeJob) {
    const pct = Math.max(0, Math.min(100, Math.round(activeJob.pct || 0)));
    if (progressBar) progressBar.style.width = `${pct}%`;
    if (progressPercent) progressPercent.textContent = `${pct}%`;
    if (statusText) {
      const st = (activeJob.status || 'PENDING').toUpperCase();
      const isErr = st === 'FAILED';
      const isDone = st === 'COMPLETED';
      statusText.classList.toggle('text-error', isErr);
      statusText.classList.toggle('text-success', isDone);

      if (isDone) {
        const allDone = jobs.every((j) => (j.status || '').toUpperCase() === 'COMPLETED');
        statusText.textContent = allDone
          ? `Indexed all ${jobs.length} document(s)`
          : `Indexed · ${activeJob.filename} (${activeJob.chunks_created || 0} chunks)`;
      } else if (isErr) {
        statusText.textContent = `Failed: ${activeJob.error_message || 'Ingestion failed'}`;
      } else if (st === 'PROCESSING') {
        const pInfo = activeJob.total_pages > 0
          ? ` (p.${activeJob.processed_pages || 0}/${activeJob.total_pages} · ${pct}%)`
          : ` · ${pct}%`;
        statusText.textContent = `Indexing · ${activeJob.filename}${pInfo}`;
      } else {
        const qPos = activeJob.queue_position ? ` #${activeJob.queue_position}` : '';
        statusText.textContent = `Queued${qPos} · ${activeJob.filename}`;
      }
    }
  }

  if (!queueList) return;
  queueList.innerHTML = '';

  jobs.forEach((j) => {
    const item = document.createElement('div');
    item.className = 'queue-item-card';

    const titleSpan = document.createElement('span');
    titleSpan.className = 'queue-item-title';
    titleSpan.title = j.filename;
    titleSpan.textContent = j.filename;

    const badge = document.createElement('span');
    badge.className = 'queue-badge';
    const st = (j.status || 'PENDING').toUpperCase();

    if (st === 'PROCESSING') {
      badge.classList.add('queue-badge-processing');
      const pInfo = j.total_pages > 0 ? ` p.${j.processed_pages || 0}/${j.total_pages}` : '';
      badge.textContent = `Indexing${pInfo}`;
    } else if (st === 'COMPLETED') {
      badge.classList.add('queue-badge-completed');
      badge.textContent = j.isDeduplicated ? 'Dedup' : 'Indexed';
    } else if (st === 'FAILED') {
      badge.classList.add('queue-badge-failed');
      badge.textContent = 'Failed';
    } else {
      badge.classList.add('queue-badge-waiting');
      const qPos = j.queue_position ? ` #${j.queue_position}` : '';
      badge.textContent = `Queued${qPos}`;
    }

    item.appendChild(titleSpan);
    item.appendChild(badge);
    queueList.appendChild(item);
  });
}

function getSelectedAllowedRoles() {
  const fieldset = document.getElementById('doc-roles');
  if (!fieldset) return 'ADMIN,MANAGER,ANALYST,VIEWER';
  const checked = Array.from(fieldset.querySelectorAll('input[type="checkbox"]:checked')).map(
    (cb) => cb.value
  );
  if (!checked.includes('ADMIN')) {
    checked.unshift('ADMIN');
  }
  return checked.join(',');
}

const MAX_POLL_ATTEMPTS = 240;
const MAX_CONSECUTIVE_POLL_ERRORS = 5;

function startJobPolling(jobId) {
  if (!jobId) return;

  // Clear any existing timer for this jobId
  if (state.pollingTimeouts[jobId]) {
    clearTimeout(state.pollingTimeouts[jobId]);
    delete state.pollingTimeouts[jobId];
  }
  if (state.pollingIntervals[jobId]) {
    clearInterval(state.pollingIntervals[jobId]);
    delete state.pollingIntervals[jobId];
  }

  const pollStartTime = Date.now();
  let pollAttempts = 0;
  let consecutiveErrors = 0;

  const pollIteration = async () => {
    pollAttempts += 1;
    if (pollAttempts > MAX_POLL_ATTEMPTS) {
      delete state.pollingTimeouts[jobId];
      if (state.uploadQueue[jobId]) {
        state.uploadQueue[jobId].status = 'FAILED';
        state.uploadQueue[jobId].error_message = 'Polling timeout: Ingestion exceeded wait window';
      }
      renderUploadQueueUI();
      return;
    }

    try {
      const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
        headers: getAuthHeaders(),
      });

      // Automatic 429 Cooldown Recovery
      if (res.status === 429) {
        const retryAfterSec = parseInt(res.headers.get('Retry-After') || '5', 10) || 5;
        show429CooldownBanner(retryAfterSec);
        state.pollingTimeouts[jobId] = setTimeout(pollIteration, (retryAfterSec + 1) * 1000);
        return;
      }

      if (!res.ok) {
        consecutiveErrors += 1;
        if (consecutiveErrors >= MAX_CONSECUTIVE_POLL_ERRORS) {
          delete state.pollingTimeouts[jobId];
          const errData = await safeParseJson(res);
          if (state.uploadQueue[jobId]) {
            state.uploadQueue[jobId].status = 'FAILED';
            state.uploadQueue[jobId].error_message = formatHttpError(res, errData, 'Status check failed');
          }
          renderUploadQueueUI();
          return;
        }
        // Retry shortly
        state.pollingTimeouts[jobId] = setTimeout(pollIteration, 2000);
        return;
      }

      consecutiveErrors = 0;
      hide429CooldownBanner();
      const job = await safeParseJson(res);
      const status = (job.status || 'PENDING').toUpperCase();

      let pct = 15;
      if (typeof job.progress === 'number') {
        pct = Math.round(job.progress * 100);
      } else if (status === 'COMPLETED') {
        pct = 100;
      } else if (job.total_pages && job.total_pages > 0) {
        pct = Math.min(95, Math.max(15, Math.round(((job.processed_pages || 0) / job.total_pages) * 100)));
      } else if (status === 'PROCESSING') {
        pct = 50;
      }

      // Update uploadQueue state
      if (!state.uploadQueue[jobId]) {
        state.uploadQueue[jobId] = { id: jobId, filename: job.filename || 'document.pdf' };
      }
      state.uploadQueue[jobId].status = status;
      state.uploadQueue[jobId].current_stage = job.current_stage || status;
      state.uploadQueue[jobId].queue_position = job.queue_position;
      state.uploadQueue[jobId].processed_pages = job.processed_pages || 0;
      state.uploadQueue[jobId].total_pages = job.total_pages || 0;
      state.uploadQueue[jobId].chunks_created = job.chunks_created || 0;
      state.uploadQueue[jobId].pct = pct;

      renderUploadQueueUI();

      if (status === 'COMPLETED') {
        delete state.pollingTimeouts[jobId];
        await loadDocumentCatalog();
        return;
      }

      if (status === 'FAILED') {
        delete state.pollingTimeouts[jobId];
        state.uploadQueue[jobId].error_message = job.error_message || 'Ingestion failed';
        renderUploadQueueUI();
        return;
      }

      // Adaptive Progressive Polling:
      // 1000ms for first 5s -> 2000ms up to 30s -> 3500ms after 30s
      const elapsedMs = Date.now() - pollStartTime;
      let nextDelayMs = 1000;
      if (elapsedMs > 30000) {
        nextDelayMs = 3500;
      } else if (elapsedMs > 5000) {
        nextDelayMs = 2000;
      }

      state.pollingTimeouts[jobId] = setTimeout(pollIteration, nextDelayMs);
    } catch (err) {
      consecutiveErrors += 1;
      console.error('Job polling error:', err);
      if (consecutiveErrors >= MAX_CONSECUTIVE_POLL_ERRORS) {
        delete state.pollingTimeouts[jobId];
        if (state.uploadQueue[jobId]) {
          state.uploadQueue[jobId].status = 'FAILED';
          state.uploadQueue[jobId].error_message = 'Connection lost during polling';
        }
        renderUploadQueueUI();
      } else {
        state.pollingTimeouts[jobId] = setTimeout(pollIteration, 2500);
      }
    }
  };

  // Schedule first poll
  state.pollingTimeouts[jobId] = setTimeout(pollIteration, 400);
}

async function handleFiles(files) {
  if (!files || !files.length) return;
  const fileList = Array.from(files);

  if (!state.currentUser || !['ADMIN', 'MANAGER'].includes((state.currentUser.role || '').toUpperCase())) {
    updateUploadProgressUI(0, 'Forbidden: ADMIN or MANAGER role required to upload.', true, false);
    return;
  }

  const pdfFiles = fileList.filter(
    (f) => f.type === 'application/pdf' || (f.name && f.name.toLowerCase().endsWith('.pdf'))
  );

  if (pdfFiles.length === 0) {
    updateUploadProgressUI(0, 'Error: Only PDF files are supported.', true, false);
    return;
  }

  const visibilityEl = document.getElementById('doc-visibility');
  const visibility = visibilityEl ? visibilityEl.value : 'ORG';
  const allowedRoles = getSelectedAllowedRoles();

  updateUploadProgressUI(5, `Queueing ${pdfFiles.length} file(s)...`, false, false);

  // Pre-seed queue items in state for immediate responsive feedback
  pdfFiles.forEach((file, idx) => {
    const tempKey = `upload_${Date.now()}_${idx}_${file.name}`;
    file._tempQueueKey = tempKey;
    state.uploadQueue[tempKey] = {
      id: tempKey,
      filename: file.name,
      status: 'PENDING',
      current_stage: 'QUEUED',
      queue_position: idx + 1,
      processed_pages: 0,
      total_pages: 0,
      chunks_created: 0,
      pct: 5,
    };
  });
  renderUploadQueueUI();

  // Sequential upload dispatch to respect concurrency=1 and prevent socket contention
  for (const file of pdfFiles) {
    const tempKey = file._tempQueueKey;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('visibility', visibility);
    formData.append('allowed_roles', allowedRoles);

    try {
      let res = await fetch('/api/upload', {
        method: 'POST',
        headers: getAuthHeaders(),
        body: formData,
      });

      // 429 Cooldown handling on upload
      if (res.status === 429) {
        const retryAfterSec = parseInt(res.headers.get('Retry-After') || '5', 10) || 5;
        show429CooldownBanner(retryAfterSec);
        await new Promise((r) => setTimeout(r, (retryAfterSec + 1) * 1000));
        res = await fetch('/api/upload', {
          method: 'POST',
          headers: getAuthHeaders(),
          body: formData,
        });
      }

      const data = await safeParseJson(res);

      if (!res.ok) {
        if (state.uploadQueue[tempKey]) {
          state.uploadQueue[tempKey].status = 'FAILED';
          state.uploadQueue[tempKey].error_message = formatHttpError(res, data, 'Upload failed');
        }
        renderUploadQueueUI();
        continue;
      }

      if (data.deduplicated === true || data.duplicate === true) {
        delete state.uploadQueue[tempKey];
        state.uploadQueue[`dedup_${file.name}`] = {
          id: `dedup_${file.name}`,
          filename: file.name,
          status: 'COMPLETED',
          current_stage: 'DEDUPLICATED',
          queue_position: null,
          processed_pages: 0,
          total_pages: 0,
          chunks_created: 0,
          pct: 100,
          isDeduplicated: true,
        };
        renderUploadQueueUI();
        await loadDocumentCatalog();
        continue;
      }

      const jobId = data.job_id || data.id;
      if (jobId) {
        delete state.uploadQueue[tempKey];
        state.uploadQueue[jobId] = {
          id: jobId,
          filename: file.name,
          status: data.status || 'PENDING',
          current_stage: data.current_stage || 'QUEUED',
          queue_position: data.queue_position || null,
          processed_pages: 0,
          total_pages: 0,
          chunks_created: 0,
          pct: 15,
        };
        renderUploadQueueUI();
        startJobPolling(jobId);
      } else {
        delete state.uploadQueue[tempKey];
        await loadDocumentCatalog();
      }
    } catch (err) {
      console.error('Upload request error for', file.name, err);
      if (state.uploadQueue[tempKey]) {
        state.uploadQueue[tempKey].status = 'FAILED';
        state.uploadQueue[tempKey].error_message = 'Connection error during upload';
      }
      renderUploadQueueUI();
    }
  }
}

async function deleteDocument(fileHash) {
  if (!fileHash) return;
  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(fileHash)}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    if (res.ok) {
      await loadDocumentCatalog();
    } else {
      const errData = await safeParseJson(res);
      updateUploadProgressUI(0, `Error: ${formatHttpError(res, errData, 'Delete failed')}`, true, false);
    }
  } catch (err) {
    console.error('Failed to delete document:', err);
    updateUploadProgressUI(0, 'Connection error while deleting document', true, false);
  }
}

async function loadDocumentCatalog() {
  const docList = document.getElementById('doc-list');
  if (!docList) return;

  try {
    const res = await fetch('/api/documents', {
      headers: getAuthHeaders(),
    });
    const countEl = document.getElementById('accessible-docs-count');
    if (!res.ok) {
      if (countEl) countEl.textContent = '0';
      docList.innerHTML = '<p class="text-dim text-sm">Unable to load document catalog.</p>';
      return;
    }

    const data = await res.json();
    const docs = Array.isArray(data.documents) ? data.documents : [];
    if (countEl) countEl.textContent = docs.length;

    if (docs.length === 0) {
      docList.innerHTML = '<p class="text-dim text-sm">No documents indexed for this tenant.</p>';
      return;
    }

    const isAdmin =
      state.currentUser && (state.currentUser.role || '').toUpperCase() === 'ADMIN';

    docList.innerHTML = '';
    docs.forEach((doc) => {
      const card = document.createElement('div');
      card.className = 'doc-card';

      const pages = doc.total_pages !== undefined ? doc.total_pages : '?';
      const chunks = doc.total_chunks !== undefined ? doc.total_chunks : 0;
      const vis = (doc.visibility || 'ORG').toUpperCase();
      const status = (doc.status || 'COMPLETED').toUpperCase();

      // Check if document is currently being indexed or queued in frontend queue
      const activeJob = Object.values(state.uploadQueue).find(
        (j) => j.filename === doc.filename || (doc.file_hash && j.file_hash === doc.file_hash)
      );

      let statusBadgeHtml = '';
      if (status === 'PROCESSING' || (activeJob && activeJob.status === 'PROCESSING')) {
        const pCurr = (activeJob ? activeJob.processed_pages : doc.processed_pages) || 0;
        const pTot = (activeJob ? activeJob.total_pages : doc.total_pages) || pages;
        statusBadgeHtml = `<span class="doc-status-badge doc-status-processing">Indexing · p.${pCurr}/${pTot}</span>`;
      } else if (status === 'PENDING' || (activeJob && activeJob.status === 'PENDING')) {
        const qPos = activeJob && activeJob.queue_position ? ` #${activeJob.queue_position}` : '';
        statusBadgeHtml = `<span class="doc-status-badge doc-status-queued">Queued${qPos}</span>`;
      }

      const infoDiv = document.createElement('div');
      infoDiv.className = 'doc-info';
      infoDiv.innerHTML = `
        <span class="doc-filename" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
        <div class="doc-meta">
          <span class="doc-meta-chip">${escapeHtml(pages)}p</span>
          <span class="doc-meta-chip">${escapeHtml(chunks)} chunks</span>
          <span class="doc-meta-chip">${escapeHtml(vis)}</span>
          ${statusBadgeHtml}
        </div>
      `;
      card.appendChild(infoDiv);

      if (isAdmin && doc.file_hash) {
        const delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.className = 'doc-delete-btn';
        delBtn.title = 'Delete document';
        delBtn.setAttribute('aria-label', `Delete ${doc.filename}`);
        delBtn.innerHTML = SVG_ICONS.trash;
        delBtn.addEventListener('click', () => deleteDocument(doc.file_hash));
        card.appendChild(delBtn);
      }

      docList.appendChild(card);
    });
  } catch (err) {
    console.error('Document catalog load error:', err);
    docList.innerHTML = '<p class="text-dim text-sm">Catalog connection error.</p>';
  }
}

// ---------------------------------------------------------------------------
// 6. RETRIEVAL MODE SELECTOR, ABSTENTION BANNERS & TELEMETRY RIBBON
// ---------------------------------------------------------------------------
function formatInlineMarkdown(escapedLine) {
  return escapedLine
    .replace(/`([^`]+)`/g, '<code class="md-code">$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, '$1<em>$2</em>');
}

function formatMarkdownToHtml(text) {
  const safeText = escapeHtml(text || '');
  const blocks = safeText.split(/\n\n+/);
  return blocks
    .map((block) => {
      const lines = block.split('\n');
      const isQuoteBlock = lines.every((l) => /^&gt;\s?/.test(l.trim()));
      if (isQuoteBlock) {
        const inner = lines
          .map((l) => formatInlineMarkdown(l.trim().replace(/^&gt;\s?/, '')))
          .join('<br/>');
        return `<blockquote class="md-quote">${inner}</blockquote>`;
      }

      const isListBlock = lines.every((l) => /^-\s+/.test(l.trim()));
      if (isListBlock) {
        const items = lines
          .map((l) => `<li>${formatInlineMarkdown(l.trim().replace(/^-\s+/, ''))}</li>`)
          .join('');
        return `<ul class="md-list">${items}</ul>`;
      }

      // Mixed block containing lines that may be quotes, bullets, or regular text
      const renderedLines = [];
      let listItems = [];
      const flushList = () => {
        if (listItems.length > 0) {
          renderedLines.push(`<ul class="md-list">${listItems.join('')}</ul>`);
          listItems = [];
        }
      };

      lines.forEach((line) => {
        const trimmed = line.trim();
        if (/^-\s+/.test(trimmed)) {
          listItems.push(`<li>${formatInlineMarkdown(trimmed.replace(/^-\s+/, ''))}</li>`);
        } else if (/^&gt;\s?/.test(trimmed)) {
          flushList();
          renderedLines.push(
            `<blockquote class="md-quote">${formatInlineMarkdown(trimmed.replace(/^&gt;\s?/, ''))}</blockquote>`
          );
        } else {
          flushList();
          renderedLines.push(`<p>${formatInlineMarkdown(line)}</p>`);
        }
      });
      flushList();
      return renderedLines.join('');
    })
    .join('');
}

function appendUserMessage(text) {
  const chatHistory = document.getElementById('chat-history');
  if (!chatHistory) return null;

  const emptyState = chatHistory.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  const msgDiv = document.createElement('div');
  msgDiv.className = 'msg msg-user';
  msgDiv.textContent = text;

  chatHistory.appendChild(msgDiv);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return msgDiv;
}

function appendTypingIndicator() {
  const chatHistory = document.getElementById('chat-history');
  if (!chatHistory) return null;

  const emptyState = chatHistory.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  const indicator = document.createElement('div');
  indicator.className = 'msg msg-ai typing';
  indicator.innerHTML = '<span></span><span></span><span></span>';
  chatHistory.appendChild(indicator);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return indicator;
}

function buildCitationsElement(citationsArray) {
  if (!Array.isArray(citationsArray) || citationsArray.length === 0) return null;

  const container = document.createElement('div');
  container.className = 'citations';

  citationsArray.forEach((cit, idx) => {
    const chip = document.createElement('span');
    chip.className = 'citation-chip';

    const pageStart = cit.page !== undefined ? cit.page : '?';
    const pageEnd = cit.page_end !== undefined && cit.page_end !== pageStart ? `–${cit.page_end}` : '';
    const pageLabel = `p.${pageStart}${pageEnd}`;
    const parentId = cit.parent_id ? String(cit.parent_id).slice(0, 10) : 'root';
    const scoreVal = typeof cit.score === 'number' ? cit.score.toFixed(3) : cit.score || '0.000';
    const snippetText = cit.snippet || 'No verbatim snippet available.';

    chip.innerHTML = `
      ${SVG_ICONS.file}
      <span>[${idx + 1}] ${escapeHtml(cit.pdf || 'Document')}</span>
      <span class="citation-page">${escapeHtml(pageLabel)}</span>
      <span class="citation-parent">#${escapeHtml(parentId)}</span>
      <span class="citation-score">${escapeHtml(scoreVal)}</span>
      <span class="citation-tooltip">${escapeHtml(snippetText)}</span>
    `;
    container.appendChild(chip);
  });

  return container;
}

function buildTelemetryRibbon(data) {
  const telemetry = data.telemetry || {};
  const totalMs = Number(telemetry.total_latency_ms || 0).toFixed(1);
  const embedMs = Number(telemetry.embed_latency_ms || 0).toFixed(1);
  const searchMs = Number(telemetry.search_latency_ms || 0).toFixed(1);
  const rerankMs = Number(telemetry.rerank_latency_ms || 0).toFixed(1);
  const llmMs = Number(telemetry.llm_latency_ms || 0).toFixed(1);

  const promptTokens = telemetry.prompt_tokens !== undefined ? telemetry.prompt_tokens : 0;
  const completionTokens = telemetry.completion_tokens !== undefined ? telemetry.completion_tokens : 0;
  const estCost = Number(telemetry.estimated_cost_usd || 0).toFixed(6);
  const modeUsed = data.mode || state.mode;

  const ribbon = document.createElement('div');
  ribbon.className = 'telemetry-ribbon';
  ribbon.innerHTML = `
    <span class="telemetry-chip telemetry-total" title="Total Execution Latency">
      ${SVG_ICONS.clock}
      Total: ${escapeHtml(totalMs)}ms
    </span>
    <span class="telemetry-chip telemetry-breakdown" title="Pipeline Stage Breakdown">
      Embed: ${escapeHtml(embedMs)}ms | Search: ${escapeHtml(searchMs)}ms | Rerank: ${escapeHtml(rerankMs)}ms | LLM: ${escapeHtml(llmMs)}ms
    </span>
    <span class="telemetry-chip">
      ${SVG_ICONS.cpu}
      ${escapeHtml(promptTokens)}p / ${escapeHtml(completionTokens)}c
    </span>
    <span class="telemetry-chip telemetry-chip-accent">
      ${SVG_ICONS.dollar}
      $${escapeHtml(estCost)}
    </span>
    <span class="telemetry-chip">
      ${SVG_ICONS.layers}
      ${escapeHtml(modeUsed)}
    </span>
  `;
  return ribbon;
}

function getStepIconSvg(stepType) {
  const normalized = (stepType || '').toLowerCase();
  if (normalized === 'multi_query') {
    return '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>';
  }
  if (normalized === 'retrieval') {
    return '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>';
  }
  if (normalized === 'crag') {
    return '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>';
  }
  if (normalized === 'tool') {
    return '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg>';
  }
  if (normalized === 'synthesis') {
    return '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>';
  }
  if (normalized === 'memory' || normalized === 'memory_persist') {
    return '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>';
  }
  return '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>';
}

function buildReasoningTraceAccordion(reasoningTrace, toolCalls) {
  const steps = Array.isArray(reasoningTrace) ? reasoningTrace : [];
  const tools = Array.isArray(toolCalls) ? toolCalls : [];
  if (steps.length === 0 && tools.length === 0) return null;

  const container = document.createElement('div');
  container.className = 'reasoning-accordion';

  const totalSteps = steps.length;
  const toggleBtn = document.createElement('button');
  toggleBtn.type = 'button';
  toggleBtn.className = 'reasoning-toggle-btn';
  toggleBtn.setAttribute('aria-expanded', 'false');
  toggleBtn.innerHTML = `
    <div class="reasoning-toggle-left">
      <svg class="reasoning-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <polyline points="12 6 12 12 16 14"></polyline>
      </svg>
      <span>Agent Reasoning Trace</span>
      <span class="reasoning-count-badge">${totalSteps} step${totalSteps === 1 ? '' : 's'}</span>
    </div>
    <div class="reasoning-toggle-right">
      <span class="reasoning-status-pill">CRAG Grounded</span>
      <svg class="reasoning-chevron" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="6 9 12 15 18 9"></polyline>
      </svg>
    </div>
  `;

  const contentBox = document.createElement('div');
  contentBox.className = 'reasoning-content hidden';

  const timeline = document.createElement('div');
  timeline.className = 'reasoning-timeline';

  steps.forEach((stepItem) => {
    const itemEl = document.createElement('div');
    itemEl.className = 'reasoning-step-item';
    const iconSvg = getStepIconSvg(stepItem.step);

    itemEl.innerHTML = `
      <div class="step-indicator">
        ${iconSvg}
      </div>
      <div class="step-details">
        <span class="step-title">${escapeHtml(stepItem.title || stepItem.step || 'Step')}</span>
        <span class="step-desc">${escapeHtml(stepItem.detail || '')}</span>
      </div>
    `;
    timeline.appendChild(itemEl);
  });

  if (tools.length > 0) {
    const toolRow = document.createElement('div');
    toolRow.className = 'reasoning-tool-row';
    tools.forEach((tc) => {
      const chip = document.createElement('span');
      chip.className = 'tool-call-chip';
      const toolName = tc.tool || 'tool';
      const toolStatus = tc.status || 'executed';
      chip.innerHTML = `
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="4 17 10 11 4 5"></polyline>
          <line x1="12" y1="19" x2="20" y2="19"></line>
        </svg>
        <strong>${escapeHtml(toolName)}</strong>: ${escapeHtml(toolStatus)}
      `;
      toolRow.appendChild(chip);
    });
    timeline.appendChild(toolRow);
  }

  contentBox.appendChild(timeline);
  container.appendChild(toggleBtn);
  container.appendChild(contentBox);

  toggleBtn.addEventListener('click', () => {
    const isExpanded = toggleBtn.classList.toggle('expanded');
    contentBox.classList.toggle('hidden', !isExpanded);
    toggleBtn.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
  });

  return container;
}

function renderQueryResponse(data) {
  const chatHistory = document.getElementById('chat-history');
  if (!chatHistory) return;

  if (data.abstained === true) {
    const reason = (data.abstention_reason || 'LOW_CONFIDENCE').toUpperCase();
    let bannerClass = 'abstention-low-confidence';
    let iconSvg = SVG_ICONS.alert;

    if (reason === 'RBAC_EMPTY_RETRIEVAL' || reason === 'EMPTY_RETRIEVAL') {
      bannerClass = 'abstention-rbac-empty';
      iconSvg = SVG_ICONS.shield;
    } else if (reason === 'LLM_ABSTAIN') {
      bannerClass = 'abstention-llm-abstain';
      iconSvg = SVG_ICONS.alert;
    }

    const topScoreText =
      data.top_score !== null && data.top_score !== undefined
        ? Number(data.top_score).toFixed(3)
        : 'N/A';

    const banner = document.createElement('div');
    banner.className = `abstention-banner ${bannerClass}`;
    banner.innerHTML = `
      ${iconSvg}
      <div class="abstention-content">
        <div><strong>Two-Tier Abstention Gate [${escapeHtml(reason)}]</strong> (Top Rerank Score: <code>${escapeHtml(topScoreText)}</code>)</div>
        <div style="margin-top: 4px;">${escapeHtml(data.answer || 'I cannot answer this based on the provided documents.')}</div>
      </div>
    `;
    const contentBox = banner.querySelector('.abstention-content');
    if (contentBox) {
      const reasoningAccordion = buildReasoningTraceAccordion(data.reasoning_trace, data.tool_calls);
      if (reasoningAccordion) {
        contentBox.appendChild(reasoningAccordion);
      }
      contentBox.appendChild(buildTelemetryRibbon(data));
    }
    chatHistory.appendChild(banner);
  } else {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg msg-ai';

    if (data.provider_fallback === true || data.provider_status === 'degraded_extractive') {
      const fallbackBadge = document.createElement('div');
      fallbackBadge.className = 'provider-notice-badge';
      fallbackBadge.innerHTML = `
        ${SVG_ICONS.alert}
        <span>Extractive Grounded Synthesis (Local LM Studio Offline)</span>
      `;
      msgDiv.appendChild(fallbackBadge);
    }

    const contentWrapper = document.createElement('div');
    contentWrapper.innerHTML = formatMarkdownToHtml(data.answer || '');
    msgDiv.appendChild(contentWrapper);

    const citationsEl = buildCitationsElement(data.citations);
    if (citationsEl) {
      msgDiv.appendChild(citationsEl);
    }

    const reasoningAccordion = buildReasoningTraceAccordion(data.reasoning_trace, data.tool_calls);
    if (reasoningAccordion) {
      msgDiv.appendChild(reasoningAccordion);
    }

    msgDiv.appendChild(buildTelemetryRibbon(data));
    chatHistory.appendChild(msgDiv);
  }

  chatHistory.scrollTop = chatHistory.scrollHeight;

  if (typeof ChatSessionController !== 'undefined' && ChatSessionController.onQueryCompleted) {
    ChatSessionController.onQueryCompleted(data);
  }
}

const ChatSessionController = {
  activeSessionId: localStorage.getItem('rag_active_session_id') || null,
  sessionTitle: 'Investigation Session',
  turnCount: 0,

  init() {
    this.wireListeners();
    this.updateHeaderUI();
  },

  wireListeners() {
    const newSessionNavBtn = document.getElementById('new-session-btn');
    const workspaceNewSessionBtn = document.getElementById('workspace-new-session-btn');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    if (newSessionNavBtn) {
      newSessionNavBtn.addEventListener('click', () => {
        this.createNewSession();
      });
    }

    if (workspaceNewSessionBtn) {
      workspaceNewSessionBtn.addEventListener('click', () => {
        this.createNewSession();
      });
    }

    if (clearChatBtn) {
      clearChatBtn.addEventListener('click', () => {
        this.resetChatHistoryUI();
        this.turnCount = 0;
        this.updateHeaderUI();
      });
    }
  },

  updateHeaderUI() {
    const activeIndicator = document.getElementById('active-session-title');
    const displayTitle = document.getElementById('session-display-title');
    const turnCounter = document.getElementById('session-turn-counter');
    const reasoningChip = document.querySelector('#reasoning-trace .reasoning-chip-label');

    const shortId = this.activeSessionId ? this.activeSessionId.slice(0, 8) : 'New';

    if (activeIndicator) {
      activeIndicator.textContent = `Session: ${shortId}`;
    }

    if (displayTitle) {
      displayTitle.textContent = `${this.sessionTitle} (${shortId})`;
    }

    if (turnCounter) {
      turnCounter.textContent = `${this.turnCount} turn${this.turnCount === 1 ? '' : 's'}`;
    }

    if (reasoningChip) {
      reasoningChip.textContent = 'Agent Reasoning Active';
    }
  },

  async syncSession() {
    if (!state.token) return;

    if (this.activeSessionId) {
      const loaded = await this.loadSession(this.activeSessionId);
      if (loaded) return;
    }

    try {
      const res = await fetch('/api/chat/sessions', {
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        const sessions = await safeParseJson(res);
        if (Array.isArray(sessions) && sessions.length > 0) {
          const first = sessions[0];
          this.sessionTitle = first.title || 'Investigation Session';
          await this.loadSession(first.id);
          return;
        }
      }
    } catch (err) {
      console.warn('Session list fetch error:', err);
    }

    await this.createNewSession('Investigation Session');
  },

  async createNewSession(customTitle = 'Investigation Session') {
    if (!state.token) return;

    try {
      const res = await fetch('/api/chat/sessions', {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ title: customTitle }),
      });

      if (res.ok) {
        const data = await safeParseJson(res);
        this.activeSessionId = data.id;
        this.sessionTitle = data.title || customTitle;
        this.turnCount = 0;
        localStorage.setItem('rag_active_session_id', data.id);
        this.resetChatHistoryUI();
        this.updateHeaderUI();
      }
    } catch (err) {
      console.error('Failed to create new session:', err);
    }
  },

  async loadSession(sessionId) {
    if (!state.token || !sessionId) return false;

    try {
      const res = await fetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`, {
        headers: getAuthHeaders(),
      });

      if (!res.ok) {
        return false;
      }

      const messages = await safeParseJson(res);
      if (Array.isArray(messages)) {
        this.activeSessionId = sessionId;
        localStorage.setItem('rag_active_session_id', sessionId);
        this.renderHistory(messages);
        this.updateHeaderUI();
        return true;
      }
      return false;
    } catch (err) {
      console.error('Failed to load session messages:', err);
      return false;
    }
  },

  renderHistory(messages) {
    const chatHistory = document.getElementById('chat-history');
    if (!chatHistory) return;

    chatHistory.innerHTML = '';

    if (!messages || messages.length === 0) {
      this.turnCount = 0;
      this.resetChatHistoryUI();
      return;
    }

    let userTurns = 0;
    messages.forEach((msg) => {
      const role = (msg.role || '').toLowerCase();
      if (role === 'user') {
        userTurns += 1;
        appendUserMessage(msg.content);
      } else if (role === 'assistant') {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg msg-ai';

        const contentWrapper = document.createElement('div');
        contentWrapper.innerHTML = formatMarkdownToHtml(msg.content || '');
        msgDiv.appendChild(contentWrapper);

        if (Array.isArray(msg.citations) && msg.citations.length > 0) {
          const citEl = buildCitationsElement(msg.citations);
          if (citEl) msgDiv.appendChild(citEl);
        }

        chatHistory.appendChild(msgDiv);
      }
    });

    this.turnCount = userTurns;
    chatHistory.scrollTop = chatHistory.scrollHeight;
  },

  resetChatHistoryUI() {
    const chatHistory = document.getElementById('chat-history');
    if (!chatHistory) return;

    chatHistory.innerHTML = `
      <div class="empty-state">
        <div class="icon-pulse">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
        </div>
        <h2>Ready to analyze</h2>
        <p>Upload a document and ask questions to begin grounded retrieval.</p>
      </div>
    `;
  },

  onQueryCompleted(data) {
    if (data && data.session_id) {
      this.activeSessionId = data.session_id;
      localStorage.setItem('rag_active_session_id', data.session_id);
    }
    this.turnCount += 1;
    this.updateHeaderUI();

    if (data && Array.isArray(data.reasoning_trace)) {
      const reasoningChip = document.querySelector('#reasoning-trace .reasoning-chip-label');
      if (reasoningChip) {
        reasoningChip.textContent = `${data.reasoning_trace.length} steps · CRAG Grounded`;
      }
    }
  },

  clearSession() {
    this.activeSessionId = null;
    this.turnCount = 0;
    this.sessionTitle = 'Investigation Session';
    localStorage.removeItem('rag_active_session_id');
    this.updateHeaderUI();
  },
};

// ---------------------------------------------------------------------------
// 7. REAL-TIME OBSERVABILITY DRAWER CONTROLLER
// ---------------------------------------------------------------------------
function setMetricText(metricKey, value) {
  const el = document.querySelector(`[data-metric="${metricKey}"]`);
  if (el) {
    el.textContent = value;
  }
}

async function loadMetrics() {
  try {
    const orgParam =
      state.currentUser && state.currentUser.org_id
        ? `?org_id=${encodeURIComponent(state.currentUser.org_id)}`
        : '';
    const res = await fetch(`/api/metrics${orgParam}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) return;

    const data = await safeParseJson(res);
    const latencies = data.latencies || {};
    const stages = ['total', 'embed', 'search', 'rerank', 'llm'];

    stages.forEach((stage) => {
      const stats = latencies[stage] || {};
      setMetricText(`${stage}_p50`, `${Number(stats.p50 || 0).toFixed(1)}ms`);
      setMetricText(`${stage}_p90`, `${Number(stats.p90 || 0).toFixed(1)}ms`);
      setMetricText(`${stage}_p95`, `${Number(stats.p95 || 0).toFixed(1)}ms`);
      setMetricText(`${stage}_avg`, `${Number(stats.avg || 0).toFixed(1)}ms`);
    });

    const tokens = data.tokens || {};
    setMetricText('tokens_prompt', String(tokens.prompt_tokens || 0));
    setMetricText('tokens_completion', String(tokens.completion_tokens || 0));
    setMetricText('tokens_total', String(tokens.total_tokens || 0));

    const costs = data.costs || {};
    const usd = Number(costs.total_estimated_cost_usd || 0).toFixed(6);
    setMetricText('estimated_cost', `$${usd}`);

    const totalQueries = data.total_queries || 0;
    const abstainedQueries = data.abstained_queries || 0;
    const rawRate = Number(data.abstention_rate || 0);
    const ratePct = rawRate <= 1 && totalQueries > 0 ? (rawRate * 100).toFixed(1) : rawRate.toFixed(1);

    setMetricText('total_queries', String(totalQueries));
    setMetricText('abstained_count', String(abstainedQueries));
    setMetricText('abstention_rate', `${ratePct}%`);
  } catch (err) {
    console.error('Failed to fetch observability metrics:', err);
  }
}

// ---------------------------------------------------------------------------
// 8. INITIALIZATION & DOM EVENT WIRING
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const modeBtns = document.querySelectorAll('#mode-selector .pill-btn[data-mode]');
  const providerBtns = document.querySelectorAll('#provider-selector .toggle-btn[data-provider]');
  const personaBtns = document.querySelectorAll('.persona-btn[data-persona]');
  const personaSwitcher = document.getElementById('persona-switcher');
  const scrollLeftBtn = document.getElementById('persona-scroll-left');
  const scrollRightBtn = document.getElementById('persona-scroll-right');
  const queryForm = document.getElementById('query-form');
  const queryInput = document.getElementById('query-input');
  const sendBtn = document.getElementById('send-btn');

  const metricsDrawer = document.getElementById('metrics-drawer');
  const toggleMetricsBtn = document.getElementById('toggle-metrics-btn');
  const closeMetricsBtn = document.getElementById('close-metrics-btn');
  const refreshMetricsBtn = document.getElementById('refresh-metrics-btn');
  const resetMetricsBtn = document.getElementById('reset-metrics-btn');
  const drawerBackdrop = metricsDrawer ? metricsDrawer.querySelector('.drawer-backdrop') : null;

  // 1. Persona Switcher Buttons & Scroll Controls
  personaBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const personaKey = btn.dataset.persona;
      if (personaKey) {
        loginPersona(personaKey);
      }
    });
  });

  if (scrollLeftBtn && personaSwitcher) {
    scrollLeftBtn.addEventListener('click', () => {
      personaSwitcher.scrollBy({ left: -200, behavior: 'smooth' });
    });
  }

  if (scrollRightBtn && personaSwitcher) {
    scrollRightBtn.addEventListener('click', () => {
      personaSwitcher.scrollBy({ left: 200, behavior: 'smooth' });
    });
  }

  // 2. Retrieval Mode Pills
  modeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      modeBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.mode = btn.dataset.mode || 'hybrid_rerank';
    });
  });

  // 3. LLM Provider Toggles
  providerBtns.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      providerBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.provider = btn.dataset.provider || 'lmstudio';
    });
  });

  // 4. Drag and Drop File Upload
  if (dropZone && fileInput) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
      dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'));
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'));
    });

    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files) {
        handleFiles(e.dataTransfer.files);
      }
    });

    dropZone.addEventListener('click', () => {
      if (!fileInput.disabled) {
        fileInput.click();
      }
    });

    fileInput.addEventListener('change', function () {
      handleFiles(this.files);
      this.value = '';
    });
  }

  // 5. Grounded Query Submission
  if (queryForm && queryInput && sendBtn) {
    queryForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = queryInput.value.trim();
      if (!text) return;

      queryInput.value = '';
      sendBtn.disabled = true;

      appendUserMessage(text);
      const indicator = appendTypingIndicator();

      try {
        const queryBody = {
          query: text,
          mode: state.mode,
          provider: state.provider,
        };
        if (state.geminiApiKey && state.geminiApiKey.trim()) {
          queryBody.api_key = state.geminiApiKey.trim();
        }
        if (ChatSessionController.activeSessionId) {
          queryBody.session_id = ChatSessionController.activeSessionId;
        }

        let res = await fetch('/api/query', {
          method: 'POST',
          headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify(queryBody),
        });

        // 404 Recovery: if session expired or not found, generate new session and retry once
        if (res.status === 404 && ChatSessionController.activeSessionId) {
          await ChatSessionController.createNewSession();
          if (ChatSessionController.activeSessionId) {
            queryBody.session_id = ChatSessionController.activeSessionId;
            res = await fetch('/api/query', {
              method: 'POST',
              headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
              body: JSON.stringify(queryBody),
            });
          }
        }

        const data = await safeParseJson(res);
        if (indicator) indicator.remove();

        if (res.ok) {
          renderQueryResponse(data);
          if (metricsDrawer && !metricsDrawer.classList.contains('hidden')) {
            await loadMetrics();
          }
        } else {
          const chatHistory = document.getElementById('chat-history');
          if (chatHistory) {
            const errDiv = document.createElement('div');
            errDiv.className = 'msg msg-ai text-error';
            errDiv.textContent = `Error: ${formatHttpError(res, data, 'Query request failed')}`;
            chatHistory.appendChild(errDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight;
          }
        }
      } catch (err) {
        if (indicator) indicator.remove();
        console.error('Query execution error:', err);
        const chatHistory = document.getElementById('chat-history');
        if (chatHistory) {
          const errDiv = document.createElement('div');
          errDiv.className = 'msg msg-ai text-error';
          errDiv.textContent = 'Connection error while querying RAG pipeline.';
          chatHistory.appendChild(errDiv);
          chatHistory.scrollTop = chatHistory.scrollHeight;
        }
      } finally {
        sendBtn.disabled = false;
        queryInput.focus();
      }
    });
  }

  // 6. Clear Chat History
  const clearChatBtn = document.getElementById('clear-chat-btn');
  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', () => {
      const chatHistory = document.getElementById('chat-history');
      if (chatHistory) {
        chatHistory.innerHTML = `
          <div class="empty-state">
            <div class="icon-pulse">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
            </div>
            <h2>Ready to analyze</h2>
            <p>Chat history cleared. Upload a document or ask questions to begin grounded retrieval.</p>
          </div>
        `;
      }
    });
  }

  // 7. Cloud Gemini API Key Configuration Controls
  const geminiInput = document.getElementById('gemini-api-key');
  const saveGeminiKeyBtn = document.getElementById('save-gemini-key-btn');
  if (geminiInput) {
    geminiInput.value = state.geminiApiKey;
    geminiInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        if (saveGeminiKeyBtn) saveGeminiKeyBtn.click();
      }
    });
  }

  if (saveGeminiKeyBtn && geminiInput) {
    saveGeminiKeyBtn.addEventListener('click', async () => {
      const key = geminiInput.value.trim();
      state.geminiApiKey = key;
      if (key) {
        localStorage.setItem('rag_gemini_api_key', key);
      } else {
        localStorage.removeItem('rag_gemini_api_key');
      }
      updateGeminiKeyStatusUI();
      await checkLlmStatus();
    });
  }

  updateGeminiKeyStatusUI();
  checkLlmStatus();

  // 8. Observability Metrics Drawer
  if (toggleMetricsBtn && metricsDrawer) {
    toggleMetricsBtn.addEventListener('click', async () => {
      const willOpen = metricsDrawer.classList.contains('hidden');
      metricsDrawer.classList.toggle('hidden');
      if (willOpen) {
        await loadMetrics();
      }
    });
  }

  if (closeMetricsBtn && metricsDrawer) {
    closeMetricsBtn.addEventListener('click', () => {
      metricsDrawer.classList.add('hidden');
    });
  }

  if (drawerBackdrop && metricsDrawer) {
    drawerBackdrop.addEventListener('click', () => {
      metricsDrawer.classList.add('hidden');
    });
  }

  if (refreshMetricsBtn) {
    refreshMetricsBtn.addEventListener('click', () => {
      loadMetrics();
    });
  }

  if (resetMetricsBtn) {
    resetMetricsBtn.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/metrics', {
          method: 'DELETE',
          headers: getAuthHeaders(),
        });
        if (res.ok) {
          await loadMetrics();
        }
      } catch (err) {
        console.error('Failed to reset observability metrics:', err);
      }
    });
  }

  // 9. Profile Drawer & User Settings Controls
  const userProfileBtn = document.getElementById('user-profile-btn');
  const profileDrawer = document.getElementById('profile-drawer');
  const closeProfileBtn = document.getElementById('close-profile-btn');
  const profileDrawerBackdrop = document.getElementById('profile-drawer-backdrop');
  const signOutBtn = document.getElementById('sign-out-btn');

  if (userProfileBtn && profileDrawer) {
    userProfileBtn.addEventListener('click', async () => {
      const willOpen = profileDrawer.classList.contains('hidden');
      profileDrawer.classList.toggle('hidden');
      if (willOpen && state.currentUser) {
        updateProfileDrawer(state.currentUser);
        await loadStorageStatus();
      }
    });
  }

  if (closeProfileBtn && profileDrawer) {
    closeProfileBtn.addEventListener('click', () => {
      profileDrawer.classList.add('hidden');
    });
  }

  if (profileDrawerBackdrop && profileDrawer) {
    profileDrawerBackdrop.addEventListener('click', () => {
      profileDrawer.classList.add('hidden');
    });
  }

  // Storage Engine Toggle Buttons
  const storageLocalBtn = document.getElementById('storage-mode-local-btn');
  const storageCloudBtn = document.getElementById('storage-mode-cloud-btn');
  const cloudStorageFields = document.getElementById('cloud-storage-fields');
  const testStorageBtn = document.getElementById('test-storage-btn');
  const saveStorageBtn = document.getElementById('save-storage-btn');
  const storageTestStatus = document.getElementById('storage-test-status');
  const cloudQdrantUrl = document.getElementById('cloud-qdrant-url');
  const cloudQdrantKey = document.getElementById('cloud-qdrant-key');
  const cloudDbUrl = document.getElementById('cloud-database-url');

  if (storageLocalBtn && storageCloudBtn) {
    storageLocalBtn.addEventListener('click', () => {
      state.storageMode = 'local';
      storageLocalBtn.classList.add('active');
      storageCloudBtn.classList.remove('active');
      if (cloudStorageFields) cloudStorageFields.classList.add('hidden');
    });

    storageCloudBtn.addEventListener('click', () => {
      state.storageMode = 'cloud';
      storageCloudBtn.classList.add('active');
      storageLocalBtn.classList.remove('active');
      if (cloudStorageFields) cloudStorageFields.classList.remove('hidden');
    });
  }

  if (testStorageBtn) {
    testStorageBtn.addEventListener('click', async () => {
      if (storageTestStatus) {
        storageTestStatus.textContent = 'Testing connection endpoints...';
        storageTestStatus.className = 'storage-test-status';
        storageTestStatus.classList.remove('hidden');
      }
      try {
        const payload = {
          cloud_qdrant_url: cloudQdrantUrl ? cloudQdrantUrl.value.trim() : '',
          cloud_qdrant_api_key: cloudQdrantKey ? cloudQdrantKey.value.trim() : '',
          cloud_database_url: cloudDbUrl ? cloudDbUrl.value.trim() : '',
        };
        const res = await fetch('/api/config/test-storage-connection', {
          method: 'POST',
          headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify(payload),
        });
        const data = await safeParseJson(res);
        if (res.ok && data.status === 'success') {
          storageTestStatus.textContent = data.message || 'Connection test successful! Storage endpoints verified.';
          storageTestStatus.className = 'storage-test-status success';
        } else {
          storageTestStatus.textContent = data.message || data.detail || 'Connection test failed. Check endpoint parameters.';
          storageTestStatus.className = 'storage-test-status error';
        }
      } catch (err) {
        console.error('Storage test error:', err);
        if (storageTestStatus) {
          storageTestStatus.textContent = 'Network error testing storage connection.';
          storageTestStatus.className = 'storage-test-status error';
        }
      }
    });
  }

  if (saveStorageBtn) {
    saveStorageBtn.addEventListener('click', async () => {
      if (storageTestStatus) {
        storageTestStatus.textContent = 'Saving storage configuration...';
        storageTestStatus.className = 'storage-test-status';
        storageTestStatus.classList.remove('hidden');
      }
      try {
        const payload = {
          storage_mode: state.storageMode || 'local',
          cloud_qdrant_url: cloudQdrantUrl ? cloudQdrantUrl.value.trim() : '',
          cloud_qdrant_api_key: cloudQdrantKey ? cloudQdrantKey.value.trim() : '',
          cloud_database_url: cloudDbUrl ? cloudDbUrl.value.trim() : '',
        };
        const res = await fetch('/api/config/storage-mode', {
          method: 'POST',
          headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify(payload),
        });
        const data = await safeParseJson(res);
        if (res.ok) {
          storageTestStatus.textContent = `Storage configuration saved (${(state.storageMode || 'local').toUpperCase()}).`;
          storageTestStatus.className = 'storage-test-status success';
        } else {
          storageTestStatus.textContent = data.detail || 'Failed to save storage mode.';
          storageTestStatus.className = 'storage-test-status error';
        }
      } catch (err) {
        console.error('Storage save error:', err);
        if (storageTestStatus) {
          storageTestStatus.textContent = 'Network error saving storage mode.';
          storageTestStatus.className = 'storage-test-status error';
        }
      }
    });
  }

  // Sign Out Button
  if (signOutBtn) {
    signOutBtn.addEventListener('click', () => {
      localStorage.removeItem('rag_access_token');
      state.token = null;
      state.currentUser = null;
      clearAllPollingIntervals();
      ChatSessionController.clearSession();
      if (profileDrawer) profileDrawer.classList.add('hidden');

      setConnectionStatus(false, 'Signed Out');
      const badge = document.getElementById('active-user-badge');
      if (badge) {
        badge.innerHTML = `
          ${SVG_ICONS.user}
          <span class="badge-email">Signed Out</span>
        `;
      }
      const avatarInitials = document.getElementById('avatar-initials');
      if (avatarInitials) avatarInitials.textContent = '--';

      const docList = document.getElementById('doc-list');
      if (docList) {
        docList.innerHTML = `
          <div class="empty-state">
            <p class="text-dim text-sm">Please sign in to view documents.</p>
          </div>
        `;
      }

      const chatHistory = document.getElementById('chat-history');
      if (chatHistory) {
        chatHistory.innerHTML = `
          <div class="empty-state">
            <div class="icon-pulse">
              ${SVG_ICONS.shield}
            </div>
            <h2>Session Ended</h2>
            <p>You have signed out. Please authenticate via Enterprise Login Gate to continue.</p>
          </div>
        `;
      }

      const overlay = document.getElementById('login-gate-overlay');
      if (overlay) overlay.classList.remove('hidden');
    });
  }

  // 10. Enterprise Login Gate & Dual-Mode Recovery Controller
  const LoginGateController = {
    overlay: document.getElementById('login-gate-overlay'),
    loginForm: document.getElementById('login-form'),
    emailInput: document.getElementById('login-email'),
    passwordInput: document.getElementById('login-password'),
    rememberCheckbox: document.getElementById('login-remember-me'),
    errorMsg: document.getElementById('login-error-msg'),

    init() {
      this.wireListeners();
      this.checkStoredSession();
    },

    show() {
      if (this.overlay) this.overlay.classList.remove('hidden');
    },

    hide() {
      if (this.overlay) this.overlay.classList.add('hidden');
      if (this.errorMsg) {
        this.errorMsg.classList.add('hidden');
        this.errorMsg.textContent = '';
      }
    },

    async checkStoredSession() {
      const savedToken = localStorage.getItem('rag_access_token');
      if (!savedToken) {
        this.show();
        return;
      }

      try {
        const res = await fetch('/api/auth/me', {
          headers: {
            'Authorization': `Bearer ${savedToken}`,
          },
        });
        if (res.ok) {
          const user = await safeParseJson(res);
          state.token = savedToken;
          state.currentUser = user;
          setConnectionStatus(true, 'Connected');
          updateActiveUserBadge(user);
          applyRolePermissions(user);
          await loadDocumentCatalog();
          await ChatSessionController.syncSession();
          this.hide();

          // Highlight persona button if matching
          Object.entries(DEMO_PERSONAS).forEach(([k, creds]) => {
            if (creds.email === user.email) {
              document.querySelectorAll('.persona-btn[data-persona]').forEach((b) => {
                b.classList.toggle('active', b.dataset.persona === k);
              });
            }
          });
          return;
        }
      } catch (err) {
        console.warn('Stored token validation error:', err);
      }

      localStorage.removeItem('rag_access_token');
      this.show();
    },

    wireListeners() {
      // Password visibility toggle
      const togglePassBtn = document.getElementById('toggle-password-visibility-btn');
      if (togglePassBtn && this.passwordInput) {
        togglePassBtn.addEventListener('click', () => {
          const isPass = this.passwordInput.type === 'password';
          this.passwordInput.type = isPass ? 'text' : 'password';
          togglePassBtn.innerHTML = isPass
            ? '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>'
            : '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>';
        });
      }

      // Quick-Fill Buttons
      const quickfillBtns = document.querySelectorAll('.quickfill-btn[data-quickfill]');
      quickfillBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
          const key = btn.dataset.quickfill;
          const creds = DEMO_PERSONAS[key];
          if (creds && this.emailInput && this.passwordInput) {
            this.emailInput.value = creds.email;
            this.passwordInput.value = creds.password;
            if (this.errorMsg) {
              this.errorMsg.classList.add('hidden');
            }
          }
        });
      });

      // Login Form Submit
      if (this.loginForm) {
        this.loginForm.addEventListener('submit', async (e) => {
          e.preventDefault();
          const email = this.emailInput ? this.emailInput.value.trim() : '';
          const password = this.passwordInput ? this.passwordInput.value : '';
          const remember = this.rememberCheckbox ? this.rememberCheckbox.checked : true;
          const submitBtn = document.getElementById('login-submit-btn');

          if (!email || !password) return;

          if (this.errorMsg) {
            this.errorMsg.classList.add('hidden');
            this.errorMsg.textContent = '';
          }
          if (submitBtn) submitBtn.disabled = true;

          try {
            let res = await fetch('/api/auth/login', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ email, password }),
            });

            if (res.status === 401) {
              const isDemo = Object.values(DEMO_PERSONAS).some((p) => p.email === email);
              if (isDemo) {
                await fetch('/api/auth/seed', { method: 'POST' });
                res = await fetch('/api/auth/login', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ email, password }),
                });
              }
            }

            const data = await safeParseJson(res);
            if (res.ok) {
              state.token = data.access_token;
              state.currentUser = data.user;

              if (remember) {
                localStorage.setItem('rag_access_token', data.access_token);
              } else {
                localStorage.removeItem('rag_access_token');
              }

              setConnectionStatus(true, 'Connected');
              updateActiveUserBadge(state.currentUser);
              applyRolePermissions(state.currentUser);
              await loadDocumentCatalog();
              await ChatSessionController.syncSession();
              this.hide();

              Object.entries(DEMO_PERSONAS).forEach(([k, creds]) => {
                if (creds.email === email) {
                  document.querySelectorAll('.persona-btn[data-persona]').forEach((b) => {
                    b.classList.toggle('active', b.dataset.persona === k);
                  });
                }
              });
            } else {
              if (this.errorMsg) {
                this.errorMsg.textContent = data.detail || 'Invalid email or password. Please try again.';
                this.errorMsg.classList.remove('hidden');
              }
            }
          } catch (err) {
            console.error('Login submit error:', err);
            if (this.errorMsg) {
              this.errorMsg.textContent = 'Network error while attempting to sign in.';
              this.errorMsg.classList.remove('hidden');
            }
          } finally {
            if (submitBtn) submitBtn.disabled = false;
          }
        });
      }

      // Forgot Password Navigation
      const forgotLink = document.getElementById('forgot-password-link');
      const backToLoginBtn = document.getElementById('back-to-login-btn');
      const loginView = document.getElementById('login-form-view');
      const forgotView = document.getElementById('forgot-password-view');

      if (forgotLink && loginView && forgotView) {
        forgotLink.addEventListener('click', () => {
          loginView.classList.add('hidden');
          forgotView.classList.remove('hidden');
          const resetEmail = document.getElementById('reset-email');
          if (resetEmail && this.emailInput && this.emailInput.value) {
            resetEmail.value = this.emailInput.value;
          }
        });
      }

      if (backToLoginBtn && loginView && forgotView) {
        backToLoginBtn.addEventListener('click', () => {
          forgotView.classList.add('hidden');
          loginView.classList.remove('hidden');
        });
      }

      // Forgot Password Token Generation
      const getResetTokenBtn = document.getElementById('generate-reset-token-btn');
      const resetEmailInput = document.getElementById('reset-email');
      const resetTokenInput = document.getElementById('reset-token-input');
      const resetStatusMsg = document.getElementById('reset-status-msg');

      if (getResetTokenBtn) {
        getResetTokenBtn.addEventListener('click', async () => {
          const email = resetEmailInput ? resetEmailInput.value.trim() : '';
          if (!email) {
            if (resetStatusMsg) {
              resetStatusMsg.textContent = 'Please enter your account email.';
              resetStatusMsg.className = 'login-msg error-msg';
              resetStatusMsg.classList.remove('hidden');
            }
            return;
          }

          try {
            getResetTokenBtn.disabled = true;
            const res = await fetch('/api/auth/forgot-password', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ email }),
            });
            const data = await safeParseJson(res);
            if (res.ok) {
              if (data.reset_token && resetTokenInput) {
                resetTokenInput.value = data.reset_token;
              }
              if (resetStatusMsg) {
                const hint = data.master_token_hint ? ` (Hint: ${data.master_token_hint})` : '';
                resetStatusMsg.textContent = `${data.message || 'Recovery token generated.'}${hint}`;
                resetStatusMsg.className = 'login-msg success-msg';
                resetStatusMsg.classList.remove('hidden');
              }
            } else {
              if (resetStatusMsg) {
                resetStatusMsg.textContent = data.detail || 'Failed to generate reset token.';
                resetStatusMsg.className = 'login-msg error-msg';
                resetStatusMsg.classList.remove('hidden');
              }
            }
          } catch (err) {
            console.error('Forgot password token error:', err);
            if (resetStatusMsg) {
              resetStatusMsg.textContent = 'Network error requesting reset token.';
              resetStatusMsg.className = 'login-msg error-msg';
              resetStatusMsg.classList.remove('hidden');
            }
          } finally {
            getResetTokenBtn.disabled = false;
          }
        });
      }

      // Reset Password Submit
      const resetSubmitBtn = document.getElementById('reset-password-submit-btn');
      const resetNewPassInput = document.getElementById('reset-new-password');

      if (resetSubmitBtn) {
        resetSubmitBtn.addEventListener('click', async () => {
          const email = resetEmailInput ? resetEmailInput.value.trim() : '';
          const token = resetTokenInput ? resetTokenInput.value.trim() : '';
          const newPassword = resetNewPassInput ? resetNewPassInput.value : '';

          if (!email || !token || !newPassword) {
            if (resetStatusMsg) {
              resetStatusMsg.textContent = 'Please fill in all reset fields.';
              resetStatusMsg.className = 'login-msg error-msg';
              resetStatusMsg.classList.remove('hidden');
            }
            return;
          }

          if (newPassword.length < 6) {
            if (resetStatusMsg) {
              resetStatusMsg.textContent = 'Password must be at least 6 characters.';
              resetStatusMsg.className = 'login-msg error-msg';
              resetStatusMsg.classList.remove('hidden');
            }
            return;
          }

          try {
            resetSubmitBtn.disabled = true;
            const res = await fetch('/api/auth/reset-password', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                email,
                reset_token: token,
                new_password: newPassword,
              }),
            });
            const data = await safeParseJson(res);
            if (res.ok) {
              if (resetStatusMsg) {
                resetStatusMsg.textContent = data.message || 'Password reset successfully! Returning to sign in...';
                resetStatusMsg.className = 'login-msg success-msg';
                resetStatusMsg.classList.remove('hidden');
              }
              if (this.emailInput) this.emailInput.value = email;
              if (this.passwordInput) this.passwordInput.value = newPassword;

              setTimeout(() => {
                if (forgotView && loginView) {
                  forgotView.classList.add('hidden');
                  loginView.classList.remove('hidden');
                }
              }, 1200);
            } else {
              if (resetStatusMsg) {
                resetStatusMsg.textContent = data.detail || 'Password reset failed.';
                resetStatusMsg.className = 'login-msg error-msg';
                resetStatusMsg.classList.remove('hidden');
              }
            }
          } catch (err) {
            console.error('Password reset submit error:', err);
            if (resetStatusMsg) {
              resetStatusMsg.textContent = 'Network error updating password.';
              resetStatusMsg.className = 'login-msg error-msg';
              resetStatusMsg.classList.remove('hidden');
            }
          } finally {
            resetSubmitBtn.disabled = false;
          }
        });
      }
    },
  };

  ChatSessionController.init();
  LoginGateController.init();
});
