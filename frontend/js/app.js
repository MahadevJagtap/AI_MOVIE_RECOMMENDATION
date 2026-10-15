/**
 * CineMatch — Modern Movie Recommendation Application
 * Vanilla JavaScript Frontend Client
 * 
 * Interacts with FastAPI backend:
 *   - GET  /health
 *   - GET  /recommend/{user_id}
 *   - GET  /candidates/{user_id}
 *   - GET  /popular
 *   - POST /feedback
 *   - GET  /metrics/ctr
 *   - GET  /metrics/offline
 */

// Application State
const state = {
  activeView: 'discover',
  activeInsightsSubtab: 'inspect',
  selectedProfileId: '198',
  activeUserId: '198',
  genreFilter: 'All Genres',
  numRecs: 9,
  modelVariant: 'wide_and_deep',
  showScores: false,
  isFetching: false,
  currentRecommendations: null,
  currentCandidates: null,
  loggedImpressionsRequestId: null,
  savedMovieIds: new Set(),
  watchlistItems: [],
  backendHealth: null,
  movieImpressionIds: new Map(), // movieId -> eventId
};

// DOM Elements Cache
const DOM = {
  navTabs: document.querySelectorAll('.nav-tab-btn'),
  views: document.querySelectorAll('.view-section'),
  statusDot: document.getElementById('status-dot'),
  statusText: document.getElementById('status-text'),
  profileSelect: document.getElementById('profile-select'),
  genreSelect: document.getElementById('genre-select'),
  countSelect: document.getElementById('count-select'),
  customUserInput: document.getElementById('custom-user-input'),
  variantSelect: document.getElementById('variant-select'),
  showScoresCheckbox: document.getElementById('show-scores-checkbox'),
  advancedToggleBtn: document.getElementById('advanced-toggle-btn'),
  advancedDrawer: document.getElementById('advanced-drawer'),
  findMoviesBtn: document.getElementById('btn-find-movies'),
  resultsContainer: document.getElementById('results-container'),
  toastContainer: document.getElementById('toast-container'),
  // Watchlist View
  watchlistContainer: document.getElementById('watchlist-container'),
  watchlistHeaderDesc: document.getElementById('watchlist-header-desc'),
  // Insights subtabs
  insightsSubtabs: document.querySelectorAll('.subtab-btn'),
  insightsContents: document.querySelectorAll('.subtab-content'),
  // Stage 1 & 2 containers
  stage1List: document.getElementById('stage1-candidates-list'),
  stage2List: document.getElementById('stage2-recommendations-list'),
  // CTR Metrics
  ctrImpressions: document.getElementById('ctr-impressions'),
  ctrClicks: document.getElementById('ctr-clicks'),
  ctrOverall: document.getElementById('ctr-overall'),
  ctrTableBody: document.getElementById('ctr-variant-tbody'),
  ctrDisclaimer: document.getElementById('ctr-disclaimer'),
  // Offline Benchmarks
  offlineTableBody: document.getElementById('offline-benchmarks-tbody'),
};

// ============================================================================
// Utilities & Helper Functions
// ============================================================================

/** Safe HTML Escaping */
function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

/** Generate standard UUID v4 */
function generateUUID() {
  if (crypto && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/** Extract clean title and 4-digit year */
function parseTitleAndYear(rawTitle) {
  if (!rawTitle) return { title: 'Unknown Title', year: '' };
  const yearMatch = rawTitle.match(/\((\d{4})\)/);
  const year = yearMatch ? yearMatch[1] : '';
  const cleanTitle = rawTitle.replace(/\s*\(\d{4}\).*/, '').trim();
  return { title: cleanTitle || rawTitle, year };
}

/** Get cinema emoji glyph from genres */
function getGenreGlyph(genres) {
  const g = (genres || '').toLowerCase();
  if (g.includes('action')) return '💥';
  if (g.includes('sci-fi')) return '🚀';
  if (g.includes('comedy')) return '🍿';
  if (g.includes('horror') || g.includes('thriller')) return '⚡';
  if (g.includes('romance') || g.includes('drama')) return '🎭';
  if (g.includes('animation')) return '🎨';
  if (g.includes('crime')) return '🕵️';
  if (g.includes('adventure') || g.includes('fantasy')) return '🗡️';
  return '🎬';
}

/** Get ambient gradient based on primary genre */
function getGenreGradient(genres) {
  const g = (genres || '').toLowerCase();
  if (g.includes('action') || g.includes('adventure')) {
    return 'linear-gradient(135deg, #450A0A 0%, #1E1B4B 100%)';
  }
  if (g.includes('sci-fi') || g.includes('fantasy')) {
    return 'linear-gradient(135deg, #172554 0%, #312E81 100%)';
  }
  if (g.includes('comedy')) {
    return 'linear-gradient(135deg, #451A03 0%, #1E293B 100%)';
  }
  if (g.includes('thriller') || g.includes('horror') || g.includes('mystery')) {
    return 'linear-gradient(135deg, #18181B 0%, #3B0764 100%)';
  }
  if (g.includes('drama') || g.includes('romance')) {
    return 'linear-gradient(135deg, #4A044E 0%, #1E1B4B 100%)';
  }
  return 'linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%)';
}

/** Show floating toast notification */
function showToast(message, icon = '🍿') {
  if (!DOM.toastContainer) return;
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `<span>${icon}</span><span>${escapeHtml(message)}</span>`;
  DOM.toastContainer.appendChild(toast);
  setTimeout(() => {
    if (toast.parentNode) toast.parentNode.removeChild(toast);
  }, 3200);
}

// ============================================================================
// API Service Calls
// ============================================================================

async function apiFetch(endpoint, options = {}) {
  try {
    const response = await fetch(endpoint, {
      ...options,
      headers: {
        'Accept': 'application/json',
        ...(options.headers || {}),
      },
    });
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errBody.detail || `HTTP ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.warn(`API Error [${endpoint}]:`, err);
    throw err;
  }
}

/** Check system health */
async function checkSystemHealth() {
  try {
    const data = await apiFetch('/health');
    state.backendHealth = data;
    DOM.statusDot.className = 'status-dot online';
    DOM.statusText.textContent = data.status === 'healthy' ? 'System Online' : 'System Degraded';
  } catch (err) {
    state.backendHealth = null;
    DOM.statusDot.className = 'status-dot offline';
    DOM.statusText.textContent = 'Backend Offline';
  }
}

/** Fetch recommendations */
async function fetchRecommendations(userId, k, variant) {
  const params = new URLSearchParams({ k: k, variant: variant });
  return await apiFetch(`/recommend/${encodeURIComponent(userId)}?${params}`);
}

/** Fetch stage-1 candidates */
async function fetchCandidates(userId, k = 100) {
  const params = new URLSearchParams({ k: k });
  return await apiFetch(`/candidates/${encodeURIComponent(userId)}?${params}`);
}

/** Log feedback event (impression or click) */
async function logFeedbackEvent(eventData) {
  try {
    return await apiFetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(eventData),
    });
  } catch (err) {
    console.warn('Feedback logging error:', err);
    return null;
  }
}

/** Fetch live CTR */
async function fetchCtrMetrics() {
  try {
    return await apiFetch('/metrics/ctr');
  } catch (err) {
    return { total_impressions: 0, total_clicks: 0, overall_ctr: 0.0, by_variant: {} };
  }
}

/** Fetch user persistent watchlist from SQLite */
async function loadUserWatchlist(userId) {
  try {
    const data = await apiFetch(`/feedback/${encodeURIComponent(userId)}`);
    if (data && Array.isArray(data.saved_movie_ids)) {
      state.savedMovieIds = new Set(data.saved_movie_ids.map(String));
      state.watchlistItems = data.watchlist || [];
    }
  } catch (err) {
    console.warn('Could not load user watchlist from backend:', err);
    state.savedMovieIds = new Set();
    state.watchlistItems = [];
  }
}

// ============================================================================
// UI Rendering & Interaction
// ============================================================================

/** Render skeleton cards during loading */
function renderSkeletons(count = 6) {
  DOM.resultsContainer.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'movie-grid';

  for (let i = 0; i < count; i++) {
    const skel = document.createElement('div');
    skel.className = 'skeleton-card shimmer';
    skel.innerHTML = `
      <div class="skeleton-poster"></div>
      <div class="skeleton-body">
        <div class="skeleton-line title"></div>
        <div class="skeleton-line tags"></div>
        <div class="skeleton-line btn"></div>
      </div>
    `;
    grid.appendChild(skel);
  }

  DOM.resultsContainer.appendChild(grid);
}

/** Render Movie Recommendations */
function renderRecommendations(recData) {
  DOM.resultsContainer.innerHTML = '';

  if (!recData || !recData.recommendations || recData.recommendations.length === 0) {
    DOM.resultsContainer.innerHTML = `
      <div class="message-box warning">
        <span>⚠️</span>
        <div>
          <strong>No matching movies found.</strong>
          <p style="margin-top: 0.3rem;">Try selecting "All Genres" or increasing the movie count above.</p>
        </div>
      </div>
    `;
    return;
  }

  // Filter by genre on client if genre is specified
  let items = recData.recommendations;
  if (state.genreFilter !== 'All Genres') {
    items = items.filter(item => 
      (item.genres || '').toLowerCase().includes(state.genreFilter.toLowerCase())
    );
  }

  // Slice to requested count
  items = items.slice(0, state.numRecs);

  if (items.length === 0) {
    DOM.resultsContainer.innerHTML = `
      <div class="message-box warning">
        <span>🍿</span>
        <div>
          <strong>No movies matched the filter "${escapeHtml(state.genreFilter)}".</strong>
          <p style="margin-top: 0.3rem;">Try switching to "All Genres" or choosing another profile.</p>
        </div>
      </div>
    `;
    return;
  }

  // Header Banner
  const banner = document.createElement('div');
  banner.className = 'results-header-banner';

  const isCold = recData.is_cold_start;
  const userText = isCold ? 'Guest (Cold-Start)' : `Viewer #${recData.user_id}`;
  const genreText = state.genreFilter !== 'All Genres' ? ` in <strong>${escapeHtml(state.genreFilter)}</strong>` : '';

  let userExplanation = '';
  if (isCold) {
    if (String(recData.user_id) === '1') {
      userExplanation = '<div style="margin-top: 0.35rem; font-size: 0.85rem; color: #fcd34d;">ℹ️ <strong>Note on User #1:</strong> User 1 is not in the active dev training split (6,141 sampled users). Cleanly serving cold-start popularity baseline.</div>';
    } else if (String(recData.user_id) === '999999') {
      userExplanation = '<div style="margin-top: 0.35rem; font-size: 0.85rem; color: #93c5fd;">ℹ️ <strong>Guest Viewer:</strong> Unregistered profile. Serving verified popularity baseline (rating volume × average stars).</div>';
    } else {
      userExplanation = `<div style="margin-top: 0.35rem; font-size: 0.85rem; color: #fcd34d;">ℹ️ <strong>Cold-Start:</strong> User #${escapeHtml(recData.user_id)} has no historical ratings in this model cohort. Serving popularity baseline.</div>`;
    }
  } else {
    userExplanation = `<div style="margin-top: 0.35rem; font-size: 0.85rem; color: #86efac;">🎯 <strong>Personalized:</strong> Ranked with Wide &amp; Deep neural model using 32 collaborative features and two-tower vector retrieval.</div>`;
  }

  banner.innerHTML = `
    <div class="results-meta">
      <div>
        Showing <strong>${items.length}</strong> recommendations for <strong>${userText}</strong>${genreText} &bull; 
        Computed in <strong>${recData.latency_ms.toFixed(1)} ms</strong>
      </div>
      ${userExplanation}
    </div>
    <div class="results-badges">
      ${recData.cached ? '<span class="pill-badge cache">⚡ Redis Cache Hit</span>' : ''}
      ${isCold ? '<span class="pill-badge cold-start">🌟 Popularity Baseline</span>' : '<span class="pill-badge" style="background: rgba(34, 197, 94, 0.15); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.3);">🧠 Neural Personalized</span>'}
    </div>
  `;
  DOM.resultsContainer.appendChild(banner);

  // Movie Card Grid
  const grid = document.createElement('div');
  grid.className = 'movie-grid';

  items.forEach(item => {
    const card = buildMovieCard(item, recData);
    grid.appendChild(card);
  });

  DOM.resultsContainer.appendChild(grid);

  // Automatically emit genuine impression feedback for displayed items
  emitImpressions(items, recData);
}

/** Build Individual Movie Card DOM (Safe from XSS) */
function buildMovieCard(item, recData) {
  const card = document.createElement('div');
  card.className = 'movie-card';

  const { title: cleanTitle, year } = parseTitleAndYear(item.title);
  const glyph = getGenreGlyph(item.genres);
  const gradient = getGenreGradient(item.genres);
  const imdbSearchUrl = `https://www.imdb.com/find?q=${encodeURIComponent(cleanTitle)}`;

  // 1. Poster Artwork Area
  const poster = document.createElement('div');
  poster.className = 'card-poster';
  poster.style.background = gradient;

  const rankBadge = document.createElement('span');
  rankBadge.className = 'badge-rank';
  rankBadge.textContent = `#${item.rank}`;
  poster.appendChild(rankBadge);

  if (year) {
    const yearBadge = document.createElement('span');
    yearBadge.className = 'badge-year';
    yearBadge.textContent = year;
    poster.appendChild(yearBadge);
  }

  const glyphEl = document.createElement('div');
  glyphEl.className = 'card-poster-glyph';
  glyphEl.textContent = glyph;
  poster.appendChild(glyphEl);

  card.appendChild(poster);

  // 2. Card Content Area
  const content = document.createElement('div');
  content.className = 'card-content';

  const titleEl = document.createElement('div');
  titleEl.className = 'card-title';
  titleEl.title = item.title;
  titleEl.textContent = cleanTitle;
  content.appendChild(titleEl);

  // Genres
  const genresEl = document.createElement('div');
  genresEl.className = 'card-genres';
  const genreList = (item.genres || '').split('|').filter(Boolean).slice(0, 3);
  genreList.forEach(g => {
    const tag = document.createElement('span');
    tag.className = 'genre-tag';
    tag.textContent = g;
    genresEl.appendChild(tag);
  });
  content.appendChild(genresEl);

  // Footer area
  const footer = document.createElement('div');
  footer.className = 'card-footer';

  const imdbLink = document.createElement('a');
  imdbLink.className = 'imdb-link';
  imdbLink.href = imdbSearchUrl;
  imdbLink.target = '_blank';
  imdbLink.rel = 'noopener noreferrer';
  imdbLink.innerHTML = '🎬 Details on IMDb &rarr;';
  footer.appendChild(imdbLink);

  // Model attribution if toggled
  if (state.showScores) {
    const scoreAttribution = document.createElement('div');
    scoreAttribution.className = 'card-score-attribution';
    let scoreText = '';
    let sourceText = escapeHtml(item.source);

    if (item.source === 'two_stage_ranking') {
      sourceText = 'Wide & Deep Neural Reranker';
      scoreText = `🎯 Match Score: ${(item.score * 100).toFixed(1)}% (Likelihood Rating >= 3.5)`;
    } else if (item.source === 'popularity_fallback' || item.source === 'popularity_baseline') {
      sourceText = 'Popularity Baseline';
      scoreText = `🌟 Score: ${Math.round(item.score).toLocaleString()} (Ratings × Stars)`;
    } else if (item.source === 'two_tower_faiss') {
      sourceText = 'FAISS Vector Sim';
      scoreText = `📐 Cosine: ${item.score.toFixed(4)}`;
    } else {
      scoreText = `Score: ${item.score.toFixed(4)}`;
    }

    scoreAttribution.innerHTML = `
      <span>Source: ${sourceText}</span>
      <span class="card-score-val">${scoreText}</span>
    `;
    footer.appendChild(scoreAttribution);
  }

  // Watch / Feedback Button (Interactive Toggle persisted to SQLite)
  const midStr = String(item.movie_id);
  const isSaved = state.savedMovieIds.has(midStr);
  const btnWatch = document.createElement('button');
  btnWatch.className = `btn-watch ${isSaved ? 'saved' : ''}`;
  btnWatch.innerHTML = isSaved ? '✓ In Watchlist' : '❤️ Interested / Watch';

  btnWatch.addEventListener('click', async () => {
    const alreadySaved = state.savedMovieIds.has(midStr);

    if (alreadySaved) {
      // Toggle OFF: Remove from persistent SQLite watchlist
      btnWatch.disabled = true;
      try {
        await apiFetch(`/feedback/${encodeURIComponent(recData.user_id)}/${encodeURIComponent(midStr)}`, {
          method: 'DELETE',
        });
        state.savedMovieIds.delete(midStr);
        state.watchlistItems = state.watchlistItems.filter(w => String(w.movie_id) !== midStr);
        btnWatch.classList.remove('saved');
        btnWatch.innerHTML = '❤️ Interested / Watch';
        showToast(`Removed '${cleanTitle}' from your watchlist.`);
        if (state.activeView === 'watchlist') renderWatchlistView();
      } catch (err) {
        showToast(`Could not remove '${cleanTitle}': ${err.message}`, '⚠️');
      } finally {
        btnWatch.disabled = false;
      }
    } else {
      // Toggle ON: Durably persist to SQLite and append to JSONL
      btnWatch.disabled = true;
      const priorImpressionId = state.movieImpressionIds.get(item.movie_id) || state.movieImpressionIds.get(midStr);

      const clickEvent = {
        event_id: generateUUID(),
        event_type: 'click',
        user_id: String(recData.user_id),
        movie_id: midStr,
        variant: recData.variant || 'wide_and_deep',
        request_id: recData.request_id || null,
        impression_id: priorImpressionId || null,
        model_version: 'v1.0',
      };

      try {
        const res = await logFeedbackEvent(clickEvent);
        if (res && (res.status === 'success' || res.persisted)) {
          state.savedMovieIds.add(midStr);
          btnWatch.classList.add('saved');
          btnWatch.innerHTML = '✓ In Watchlist';
          showToast(`Added '${cleanTitle}' to your watchlist!`);
          updateCtrView();
          // Reload watchlist in background
          loadUserWatchlist(recData.user_id);
        } else {
          throw new Error('Database write could not be confirmed.');
        }
      } catch (err) {
        showToast(`Failed to save feedback: ${err.message}`, '⚠️');
      } finally {
        btnWatch.disabled = false;
      }
    }
  });

  footer.appendChild(btnWatch);
  content.appendChild(footer);
  card.appendChild(content);

  return card;
}

/** Emit genuine impressions for displayed items */
function emitImpressions(items, recData) {
  if (!recData || !recData.request_id) return;
  if (state.loggedImpressionsRequestId === recData.request_id) return;

  state.loggedImpressionsRequestId = recData.request_id;

  items.forEach(item => {
    const impId = generateUUID();
    state.movieImpressionIds.set(item.movie_id, impId);

    const impEvent = {
      event_id: impId,
      event_type: 'impression',
      user_id: String(recData.user_id),
      movie_id: item.movie_id,
      variant: recData.variant || 'wide_and_deep',
      request_id: recData.request_id,
      model_version: 'v1.0',
    };

    logFeedbackEvent(impEvent).catch(() => {});
  });
}

/** Render Watchlist Grid from Persistent SQLite Data */
function renderWatchlistView() {
  if (!DOM.watchlistContainer) return;
  DOM.watchlistContainer.innerHTML = '';

  const userDisplayName = state.selectedProfileId === '999999' ? 'Guest' : `Viewer #${state.activeUserId}`;
  if (DOM.watchlistHeaderDesc) {
    DOM.watchlistHeaderDesc.innerHTML = `
      Saved movies for <strong>${userDisplayName}</strong> &bull; 
      <strong>${state.watchlistItems.length}</strong> items persisted in SQLite.
    `;
  }

  if (!state.watchlistItems || state.watchlistItems.length === 0) {
    DOM.watchlistContainer.innerHTML = `
      <div class="message-box" style="margin-top: 1rem;">
        <span style="font-size: 2rem;">🍿</span>
        <div>
          <strong style="font-size: 1.05rem; color: var(--text-primary);">Your Watchlist is currently empty.</strong>
          <p style="margin-top: 0.35rem; color: var(--text-secondary); line-height: 1.5;">
            Switch to the <strong>Discover</strong> tab and click <em>"❤️ Interested / Watch"</em> on any movie card to save it here.
            Saved titles are stored permanently in the backend SQLite database across page reloads and server restarts.
          </p>
        </div>
      </div>
    `;
    return;
  }

  const grid = document.createElement('div');
  grid.className = 'movie-grid';

  state.watchlistItems.forEach(item => {
    const card = document.createElement('div');
    card.className = 'movie-card';

    const { title: cleanTitle, year } = parseTitleAndYear(item.title);
    const glyph = getGenreGlyph(item.genres);
    const gradient = getGenreGradient(item.genres);
    const imdbSearchUrl = `https://www.imdb.com/find?q=${encodeURIComponent(cleanTitle)}`;

    // Poster
    const poster = document.createElement('div');
    poster.className = 'card-poster';
    poster.style.background = gradient;
    if (year) {
      const yb = document.createElement('span');
      yb.className = 'badge-year';
      yb.textContent = year;
      poster.appendChild(yb);
    }
    const glyphEl = document.createElement('div');
    glyphEl.className = 'card-poster-glyph';
    glyphEl.textContent = glyph;
    poster.appendChild(glyphEl);
    card.appendChild(poster);

    // Content
    const content = document.createElement('div');
    content.className = 'card-content';

    const titleEl = document.createElement('div');
    titleEl.className = 'card-title';
    titleEl.title = item.title;
    titleEl.textContent = cleanTitle;
    content.appendChild(titleEl);

    // Genres
    const genresEl = document.createElement('div');
    genresEl.className = 'card-genres';
    (item.genres || '').split('|').filter(Boolean).slice(0, 3).forEach(g => {
      const tag = document.createElement('span');
      tag.className = 'genre-tag';
      tag.textContent = g;
      genresEl.appendChild(tag);
    });
    content.appendChild(genresEl);

    // Footer
    const footer = document.createElement('div');
    footer.className = 'card-footer';

    const imdbLink = document.createElement('a');
    imdbLink.className = 'imdb-link';
    imdbLink.href = imdbSearchUrl;
    imdbLink.target = '_blank';
    imdbLink.rel = 'noopener noreferrer';
    imdbLink.innerHTML = '🎬 IMDb &rarr;';
    footer.appendChild(imdbLink);

    const btnRemove = document.createElement('button');
    btnRemove.className = 'btn-watch saved';
    btnRemove.innerHTML = '🗑️ Remove';
    btnRemove.title = 'Remove from persistent watchlist';

    btnRemove.addEventListener('click', async () => {
      btnRemove.disabled = true;
      try {
        await apiFetch(`/feedback/${encodeURIComponent(state.activeUserId)}/${encodeURIComponent(item.movie_id)}`, {
          method: 'DELETE',
        });
        state.savedMovieIds.delete(String(item.movie_id));
        state.watchlistItems = state.watchlistItems.filter(w => String(w.movie_id) !== String(item.movie_id));
        showToast(`Removed '${cleanTitle}' from watchlist.`);
        renderWatchlistView();
        if (state.currentRecommendations) {
          renderRecommendations(state.currentRecommendations);
        }
      } catch (err) {
        showToast(`Error removing movie: ${err.message}`, '⚠️');
        btnRemove.disabled = false;
      }
    });

    footer.appendChild(btnRemove);
    content.appendChild(footer);
    card.appendChild(content);

    grid.appendChild(card);
  });

  DOM.watchlistContainer.appendChild(grid);
}

/** Update Stage 1 vs Stage 2 Pipeline Inspector */
function updatePipelineInspector() {
  if (!DOM.stage1List || !DOM.stage2List) return;

  // Stage 1: Candidate Pool
  DOM.stage1List.innerHTML = '';
  if (state.currentCandidates && state.currentCandidates.candidates) {
    const topCands = state.currentCandidates.candidates.slice(0, 8);
    topCands.forEach(c => {
      const el = document.createElement('div');
      el.className = 'pipeline-item stage1';
      el.innerHTML = `
        <div class="pipeline-item-top">
          <span class="pipeline-item-title">#${c.rank} ${escapeHtml(c.title)}</span>
          <span class="pipeline-item-score">Sim: ${c.retrieval_score.toFixed(4)}</span>
        </div>
        <div class="pipeline-item-genres">${escapeHtml(c.genres)}</div>
      `;
      DOM.stage1List.appendChild(el);
    });
  } else {
    DOM.stage1List.innerHTML = '<div style="color: var(--text-muted); font-size: 0.88rem; padding: 1rem 0;">No candidate data. Click "Find My Movies" on the Discovery tab.</div>';
  }

  // Stage 2: Reranked
  DOM.stage2List.innerHTML = '';
  if (state.currentRecommendations && state.currentRecommendations.recommendations) {
    const topRecs = state.currentRecommendations.recommendations.slice(0, 8);
    topRecs.forEach(r => {
      const el = document.createElement('div');
      el.className = 'pipeline-item stage2';
      let scoreBadge = '';
      if (r.source === 'two_stage_ranking') {
        scoreBadge = `Match: ${(r.score * 100).toFixed(1)}%`;
      } else if (r.source === 'popularity_fallback' || r.source === 'popularity_baseline') {
        scoreBadge = `Pop: ${Math.round(r.score).toLocaleString()}`;
      } else {
        scoreBadge = `Score: ${r.score.toFixed(4)}`;
      }
      el.innerHTML = `
        <div class="pipeline-item-top">
          <span class="pipeline-item-title">#${r.rank} ${escapeHtml(r.title)}</span>
          <span class="pipeline-item-score">${scoreBadge}</span>
        </div>
        <div class="pipeline-item-genres">${escapeHtml(r.genres)}</div>
      `;
      DOM.stage2List.appendChild(el);
    });
  } else {
    DOM.stage2List.innerHTML = '<div style="color: var(--text-muted); font-size: 0.88rem; padding: 1rem 0;">No recommendation data available.</div>';
  }
}

/** Update Live CTR View */
async function updateCtrView() {
  const data = await fetchCtrMetrics();
  if (DOM.ctrImpressions) DOM.ctrImpressions.textContent = (data.total_impressions || 0).toLocaleString();
  if (DOM.ctrClicks) DOM.ctrClicks.textContent = (data.total_clicks || 0).toLocaleString();
  if (DOM.ctrOverall) {
    const pct = ((data.overall_ctr || 0.0) * 100.0).toFixed(2);
    DOM.ctrOverall.textContent = `${pct}%`;
  }

  if (DOM.ctrDisclaimer && data.disclaimer) {
    DOM.ctrDisclaimer.textContent = data.disclaimer;
  }

  if (DOM.ctrTableBody) {
    DOM.ctrTableBody.innerHTML = '';
    const byVariant = data.by_variant || {};
    const variants = Object.keys(byVariant);
    if (variants.length === 0) {
      DOM.ctrTableBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No interaction events recorded in this session yet.</td></tr>';
    } else {
      variants.forEach(v => {
        const row = byVariant[v];
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${escapeHtml(v)}</strong></td>
          <td>${(row.impressions || 0).toLocaleString()}</td>
          <td>${(row.clicks || 0).toLocaleString()}</td>
          <td style="color: #6ee7b7; font-weight: 700;">${((row.ctr || 0) * 100).toFixed(2)}%</td>
        `;
        DOM.ctrTableBody.appendChild(tr);
      });
    }
  }
}

/** Update Offline Benchmarks View */
async function updateOfflineBenchmarksView() {
  if (!DOM.offlineTableBody) return;
  const data = await fetchOfflineMetrics();
  if (!data || !data.evaluation_results || !data.evaluation_results.models) {
    DOM.offlineTableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">Offline experiment metrics not found.</td></tr>';
    return;
  }

  DOM.offlineTableBody.innerHTML = '';
  const models = data.evaluation_results.models;
  Object.keys(models).forEach(name => {
    const m = models[name];
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${escapeHtml(name)}</strong></td>
      <td>${m['ndcg@10'] ? m['ndcg@10'].toFixed(4) : '-'}</td>
      <td>${m['recall@10'] ? m['recall@10'].toFixed(4) : '-'}</td>
      <td>${m['precision@10'] ? m['precision@10'].toFixed(4) : '-'}</td>
      <td>${m['mrr@10'] ? m['mrr@10'].toFixed(4) : '-'}</td>
      <td>${m['latency_p95_ms'] ? m['latency_p95_ms'].toFixed(2) + ' ms' : '-'}</td>
      <td>${m['unique_items_recommended@20'] ? m['unique_items_recommended@20'].toLocaleString() : '-'}</td>
    `;
    DOM.offlineTableBody.appendChild(tr);
  });
}

// ============================================================================
// Recommendation Flow Execution
// ============================================================================

async function executeRecommendationQuery() {
  if (state.isFetching) return;

  // Resolve target user ID
  const customUser = (DOM.customUserInput.value || '').trim();
  state.activeUserId = customUser || state.selectedProfileId;
  state.genreFilter = DOM.genreSelect.value;
  state.numRecs = parseInt(DOM.countSelect.value, 10) || 9;
  state.modelVariant = DOM.variantSelect.value;
  state.showScores = DOM.showScoresCheckbox.checked;

  state.isFetching = true;
  DOM.findMoviesBtn.disabled = true;
  DOM.findMoviesBtn.innerHTML = '<span>⏳</span> Finding Movies...';

  // Render Skeleton Loader
  renderSkeletons(state.numRecs);

  try {
    // Ensure active user's persistent watchlist is loaded from SQLite
    await loadUserWatchlist(state.activeUserId);

    // If genre filtering is applied, request 3x candidates to ensure sufficient filtered output
    const fetchK = state.genreFilter !== 'All Genres' ? state.numRecs * 3 : state.numRecs;

    const [recData, candData] = await Promise.all([
      fetchRecommendations(state.activeUserId, fetchK, state.modelVariant),
      fetchCandidates(state.activeUserId, 100).catch(() => null),
    ]);

    state.currentRecommendations = recData;
    state.currentCandidates = candData;

    renderRecommendations(recData);
    updatePipelineInspector();
  } catch (err) {
    console.error('Error fetching recommendations:', err);
    DOM.resultsContainer.innerHTML = `
      <div class="message-box error">
        <span>❌</span>
        <div>
          <strong>Unable to load recommendations.</strong>
          <p style="margin-top: 0.35rem;">
            ${escapeHtml(err.message || 'Please check that the FastAPI service is running on http://127.0.0.1:8000.')}
          </p>
          <button class="btn-primary" style="margin-top: 0.8rem; height: 36px; font-size: 0.85rem;" onclick="executeRecommendationQuery()">
            🔄 Try Again
          </button>
        </div>
      </div>
    `;
  } finally {
    state.isFetching = false;
    DOM.findMoviesBtn.disabled = false;
    DOM.findMoviesBtn.innerHTML = '<span>✨</span> Find My Movies';
  }
}

// ============================================================================
// Event Listeners & Initialization
// ============================================================================

function initEventListeners() {
  // Navigation Tabs Switching
  DOM.navTabs.forEach(tab => {
    tab.addEventListener('click', async () => {
      const targetView = tab.dataset.view;
      if (!targetView) return;

      DOM.navTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      DOM.views.forEach(v => {
        v.classList.toggle('active', v.id === `${targetView}-view`);
      });

      state.activeView = targetView;

      // Refresh data on switching to watchlist
      if (targetView === 'watchlist') {
        await loadUserWatchlist(state.activeUserId);
        renderWatchlistView();
      }

      // Refresh data on switching to insights
      if (targetView === 'insights') {
        updatePipelineInspector();
        updateCtrView();
        updateOfflineBenchmarksView();
      }
    });
  });

  // Insights Subtabs
  DOM.insightsSubtabs.forEach(subtab => {
    subtab.addEventListener('click', () => {
      const target = subtab.dataset.subtab;
      DOM.insightsSubtabs.forEach(s => s.classList.remove('active'));
      subtab.classList.add('active');

      DOM.insightsContents.forEach(c => {
        c.classList.toggle('active', c.id === `subtab-${target}`);
      });

      if (target === 'ctr') updateCtrView();
      if (target === 'offline') updateOfflineBenchmarksView();
    });
  });

  // Profile Selector
  DOM.profileSelect.addEventListener('change', async e => {
    state.selectedProfileId = e.target.value;
    if (!DOM.customUserInput.value.trim()) {
      await executeRecommendationQuery();
    }
  });

  // Genre Selector
  DOM.genreSelect.addEventListener('change', () => {
    executeRecommendationQuery();
  });

  // Count Selector
  DOM.countSelect.addEventListener('change', () => {
    executeRecommendationQuery();
  });

  // Advanced Drawer Toggle
  DOM.advancedToggleBtn.addEventListener('click', () => {
    DOM.advancedDrawer.classList.toggle('open');
    const isOpen = DOM.advancedDrawer.classList.contains('open');
    DOM.advancedToggleBtn.innerHTML = `${isOpen ? '▼' : '▶'} ⚙️ Advanced Preferences & Scoring Options`;
  });

  // Show Scores Toggle
  DOM.showScoresCheckbox.addEventListener('change', e => {
    state.showScores = e.target.checked;
    if (state.currentRecommendations) {
      renderRecommendations(state.currentRecommendations);
    }
  });

  // Variant Selector
  DOM.variantSelect.addEventListener('change', () => {
    executeRecommendationQuery();
  });

  // Find Movies Button
  DOM.findMoviesBtn.addEventListener('click', () => {
    executeRecommendationQuery();
  });

  // Custom User Input (Enter key)
  DOM.customUserInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      executeRecommendationQuery();
    }
  });
}

// Startup Initialization
document.addEventListener('DOMContentLoaded', async () => {
  initEventListeners();
  await checkSystemHealth();
  // Automatically trigger initial load for default user (Alex, #198)
  await loadUserWatchlist(state.activeUserId);
  executeRecommendationQuery();
});

