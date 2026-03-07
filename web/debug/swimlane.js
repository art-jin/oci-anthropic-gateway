/**
 * Swimlane Timeline Visualization
 *
 * Renders a real-time swimlane diagram for debugging gateway messages.
 * Supports both single-session and all-sessions view.
 */

class SwimlaneTimeline {
  /**
   * @param {string|null} sessionId - Session ID to visualize, or null for all sessions
   * @param {string} containerId - ID of container element for SVG
   * @param {string} detailPanelId - ID of detail panel element
   */
  constructor(sessionId, containerId, detailPanelId) {
    this.sessionId = sessionId;  // null means all sessions
    this.container = document.getElementById(containerId);
    this.detailPanel = document.getElementById(detailPanelId);
    this.events = [];
    this.eventSource = null;
    this.svg = null;
    this.defs = null;
    this.autoScroll = true;
    this.selectedEventId = null;
    this.onEventCount = null;  // Callback for event count updates

    // Layout constants
    this.laneWidth = 0;
    this.nodeRadius = 8;
    this.nodeSpacingY = 30;
    this.headerHeight = 50;
    this.paddingX = 40;

    // Lane positions (x coordinates as percentages)
    this.lanePositions = {
      client: 0.167,   // 1/6
      gateway: 0.5,    // 3/6
      oci: 0.833       // 5/6
    };

    // Session colors for distinguishing different sessions
    this.sessionColors = [
      '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
      '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#6366f1'
    ];
    this.sessionColorMap = {};
    this.markerColorMap = {};

    // ResizeObserver for container size changes
    this.resizeObserver = null;
  }

  /**
   * Get color for a session
   */
  getSessionColor(sessionId) {
    if (!sessionId) return '#6b7280';
    if (!this.sessionColorMap[sessionId]) {
      const usedColors = Object.values(this.sessionColorMap);
      const availableColor = this.sessionColors.find(c => !usedColors.includes(c)) || '#6b7280';
      this.sessionColorMap[sessionId] = availableColor;
    }
    return this.sessionColorMap[sessionId];
  }

  /**
   * Initialize: load history and connect SSE
   */
  async init() {
    // Create SVG structure
    this.createSVGStructure();

    // Load historical events
    try {
      const endpoint = this.sessionId
        ? `/sessions/${encodeURIComponent(this.sessionId)}/timeline?limit=200`
        : `/timeline?limit=500`;
      const history = await api(endpoint);
      this.events = history.events || [];
      this.render();
      this.updateEventCount();
      this.updateConnectionStatus('connecting');
    } catch (err) {
      this.showError(`Failed to load timeline: ${err.message}`);
      return;
    }

    // Connect SSE for real-time updates
    this.connectSSE();

    // Monitor scroll for auto-scroll behavior
    this.container.addEventListener('scroll', () => {
      const { scrollTop, scrollHeight, clientHeight } = this.container;
      this.autoScroll = scrollTop + clientHeight >= scrollHeight - 50;
    });

    // Handle window resize
    window.addEventListener('resize', () => this.handleResize());

    // Setup ResizeObserver to handle container size changes
    this.setupResizeObserver();
  }

  /**
   * Setup ResizeObserver to monitor container size changes
   * and re-render when width becomes valid or changes significantly
   */
  setupResizeObserver() {
    if (typeof ResizeObserver === 'undefined') {
      // Fallback for browsers without ResizeObserver support
      return;
    }

    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const newWidth = entry.contentRect?.width || this.container.clientWidth;

        // Only react to valid width changes
        if (newWidth > 0) {
          const widthChanged = this.laneWidth !== newWidth;
          const wasInvalid = !this.laneWidth || this.laneWidth <= 0;

          if (widthChanged || wasInvalid) {
            const oldWidth = this.laneWidth;
            this.laneWidth = newWidth;

            // Re-render if we have events and width was previously invalid or changed
            if (this.events.length > 0) {
              console.log('ResizeObserver: re-rendering with new width', {
                oldWidth: oldWidth,
                newWidth: newWidth,
                wasInvalid: wasInvalid
              });
              this.render();
            }
          }
        }
      }
    });

    this.resizeObserver.observe(this.container);
  }

  /**
   * Update event count display
   */
  updateEventCount() {
    if (this.onEventCount) {
      this.onEventCount(this.events.length);
    }
  }

  /**
   * Create base SVG structure with defs and lane headers
   */
  createSVGStructure() {
    this.container.innerHTML = '';

    // Lane headers (HTML overlay)
    const headers = document.createElement('div');
    headers.className = 'lane-headers';
    headers.innerHTML = `
      <div class="lane-header client">Anthropic Client</div>
      <div class="lane-header gateway">Gateway</div>
      <div class="lane-header oci">OCI GenAI</div>
    `;
    this.container.appendChild(headers);

    // SVG element
    this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.svg.classList.add('timeline-svg');
    this.container.appendChild(this.svg);

    // SVG defs (markers, gradients)
    this.defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    this.svg.appendChild(this.defs);
  }

  /**
   * Create/get a solid arrow marker using the given color.
   */
  getArrowMarkerId(color) {
    if (this.markerColorMap[color]) return this.markerColorMap[color];

    const markerId = `arrow-${color.replace('#', '')}`;
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    marker.setAttribute('id', markerId);
    marker.setAttribute('viewBox', '0 0 14 10');
    marker.setAttribute('markerWidth', '14');
    marker.setAttribute('markerHeight', '10');
    marker.setAttribute('refX', '12');
    marker.setAttribute('refY', '5');
    marker.setAttribute('orient', 'auto');
    marker.setAttribute('markerUnits', 'userSpaceOnUse');

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M 0 0 L 14 5 L 0 10 z');
    path.classList.add('arrow-marker');
    path.style.fill = color;
    path.style.stroke = color;
    marker.appendChild(path);
    this.defs.appendChild(marker);

    this.markerColorMap[color] = markerId;
    return markerId;
  }

  /**
   * Connect to SSE endpoint for real-time updates
   */
  connectSSE() {
    let url = this.sessionId
      ? `/debug/api/sessions/${encodeURIComponent(this.sessionId)}/events`
      : '/debug/api/events';
    const token = typeof window.getDebugAuthToken === 'function' ? window.getDebugAuthToken() : '';
    if (token) {
      const sep = url.includes('?') ? '&' : '?';
      url = `${url}${sep}access_token=${encodeURIComponent(token)}`;
    }

    try {
      this.eventSource = new EventSource(url);
    } catch (err) {
      this.updateConnectionStatus('disconnected');
      this.scheduleReconnect();
      return;
    }

    this.eventSource.addEventListener('connected', (e) => {
      this.updateConnectionStatus('connected');
      console.log('SSE connected:', JSON.parse(e.data));
    });

    this.eventSource.addEventListener('timeline_event', (e) => {
      try {
        const event = JSON.parse(e.data);
        this.onNewEvent(event);
      } catch (err) {
        console.error('Failed to parse SSE event:', err);
      }
    });

    this.eventSource.onerror = () => {
      this.updateConnectionStatus('disconnected');
      console.error('SSE error');
      this.scheduleReconnect();
    };
  }

  /**
   * Schedule SSE reconnection
   */
  scheduleReconnect() {
    setTimeout(() => {
      if (this.eventSource) {
        this.eventSource.close();
        this.eventSource = null;
      }
      this.updateConnectionStatus('connecting');
      this.connectSSE();
    }, 3000);
  }

  /**
   * Update connection status indicator
   */
  updateConnectionStatus(status) {
    const dot = document.getElementById('connectionDot');
    const text = document.getElementById('connectionText');
    if (!dot || !text) return;

    dot.className = 'status-dot ' + status;
    text.textContent = status === 'connected' ? 'Connected' :
                       status === 'connecting' ? 'Connecting...' : 'Disconnected';
  }

  /**
   * Handle new event from SSE
   */
  onNewEvent(event) {
    // Avoid duplicates
    if (this.events.find(e => e.id === event.id)) return;

    // Debug: log lane values for request events
    if (event.kind === 'request_summary') {
      console.log('[DEBUG] request_summary event:', {
        lane: event.lane,
        target_lane: event.target_lane,
        laneX: this.getLaneX(event.lane),
        targetX: event.target_lane ? this.getLaneX(event.target_lane) : null
      });
    }

    // Add to events list and sort by timestamp
    this.events.push(event);
    this.events.sort((a, b) => {
      // Sort by timestamp ascending
      const tsA = a.ts || '';
      const tsB = b.ts || '';
      return tsA.localeCompare(tsB);
    });
    this.updateEventCount();

    // Re-render all events to ensure correct order
    this.render();

    // Auto-scroll if enabled
    if (this.autoScroll) {
      this.scrollToBottom();
    }

    // Update status bar
    this.showStatus(`New: ${event.label}`);
  }

  /**
   * Full render of all events
   */
  render() {
    if (this.events.length === 0) {
      this.showEmpty();
      return;
    }

    // Calculate SVG height
    const height = this.headerHeight + this.events.length * this.nodeSpacingY + 50;
    this.svg.setAttribute('height', height);

    // Get container width for lane calculations
    const containerWidth = this.container.clientWidth || 800;
    this.laneWidth = containerWidth;

    // Draw lane dividers
    this.drawLaneDividers(containerWidth, height);

    // Draw all events
    const eventsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    eventsGroup.classList.add('events-layer');

    this.events.forEach((event, index) => {
      const y = this.calculateY(index);
      this.renderEvent(event, y, eventsGroup, index);
    });

    // Remove old events layer if exists
    const oldLayer = this.svg.querySelector('.events-layer');
    if (oldLayer) oldLayer.remove();

    this.svg.appendChild(eventsGroup);

    // Auto scroll to bottom
    if (this.autoScroll) {
      setTimeout(() => this.scrollToBottom(), 100);
    }
  }

  /**
   * Draw vertical lane divider lines
   */
  drawLaneDividers(width, height) {
    // Remove existing dividers
    this.svg.querySelectorAll('.lane-divider').forEach(el => el.remove());

    const x1 = width * 0.333;
    const x2 = width * 0.667;

    [x1, x2].forEach(x => {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.classList.add('lane-divider');
      line.setAttribute('x1', x);
      line.setAttribute('y1', 0);
      line.setAttribute('x2', x);
      line.setAttribute('y2', height);
      this.svg.appendChild(line);
    });
  }

  /**
   * Check if event is an internal processing event (no message transfer)
   */
  isInternalEvent(kind) {
    return kind.includes('tool_detection') || kind === 'raw_text';
  }

  /**
   * Render a single event with horizontal connector line
   */
  renderEvent(event, y, container, index) {
    const laneX = this.getLaneX(event.lane);
    const effectiveSessionId = event.session_id || this.sessionId;
    const sessionColor = this.getSessionColor(effectiveSessionId);
    const isInternal = this.isInternalEvent(event.kind);

    // Create group for this event
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.classList.add('event-node');
    if (isInternal) g.classList.add('internal');
    if (event.id === this.selectedEventId) g.classList.add('selected');
    g.dataset.eventId = event.id;
    g.dataset.dumpId = event.dump_id;
    g.dataset.index = index;

    // Draw HORIZONTAL connector line from source lane to target lane (only for non-internal events)
    if (event.target_lane && !isInternal) {
      const targetX = this.getLaneX(event.target_lane);
      const isRightward = targetX > laneX;
      const dir = isRightward ? 1 : -1;
      const sourceR = this.nodeRadius * 0.78;
      const targetR = sourceR * 1.18;

      const x1 = laneX + dir * sourceR;
      const x2 = targetX - dir * targetR;

      // Debug: log arrow coordinates
      console.log('[DEBUG] Drawing arrow for', event.kind, ':', {
        x1: Math.round(x1),
        x2: Math.round(x2),
        y: Math.round(y),
        laneX: Math.round(laneX),
        targetX: Math.round(targetX)
      });

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.classList.add('connector');

      line.setAttribute('x1', x1);
      line.setAttribute('y1', y);
      line.setAttribute('x2', x2);
      line.setAttribute('y2', y);  // Horizontal line (same y)
      line.style.stroke = sessionColor;
      line.setAttribute('marker-end', `url(#${this.getArrowMarkerId(sessionColor)})`);
      g.appendChild(line);
    }

    // Node shape: circle for message transfer, rectangle for internal processing
    if (isInternal) {
      // Internal processing: rounded rectangle
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.classList.add('node', 'internal-node', this.sanitizeClass(event.kind));
      const width = this.nodeRadius * 2.2;
      const height = this.nodeRadius * 1.4;
      rect.setAttribute('x', laneX - width / 2);
      rect.setAttribute('y', y - height / 2);
      rect.setAttribute('width', width);
      rect.setAttribute('height', height);
      rect.setAttribute('rx', 3);
      rect.style.fill = sessionColor;
      rect.style.opacity = '0.8';
      g.appendChild(rect);
    } else {
      // Message transfer: circle
      const sourceR = this.nodeRadius * 0.78;
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.classList.add('node', this.sanitizeClass(event.kind));
      circle.setAttribute('cx', laneX);
      circle.setAttribute('cy', y);
      circle.setAttribute('r', sourceR);
      circle.style.fill = '#ffffff';
      circle.style.stroke = sessionColor;
      circle.style.strokeWidth = '2.4';
      g.appendChild(circle);

      // Small circle at target position if has target_lane
      if (event.target_lane) {
        const targetX = this.getLaneX(event.target_lane);
        const targetR = sourceR * 1.18;
        const targetCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        targetCircle.classList.add('node', 'target', this.sanitizeClass(event.kind));
        targetCircle.setAttribute('cx', targetX);
        targetCircle.setAttribute('cy', y);
        targetCircle.setAttribute('r', targetR);
        targetCircle.style.fill = sessionColor;
        targetCircle.style.stroke = sessionColor;
        targetCircle.style.strokeWidth = '1.2';
        g.appendChild(targetCircle);
      }
    }

    // Label (positioned to not overlap with line)
    const labelX = event.target_lane && !isInternal && this.getLaneX(event.target_lane) > laneX
      ? this.getLaneX(event.target_lane) + 8
      : laneX + this.nodeRadius + 8;
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.classList.add('label');
    text.setAttribute('x', labelX);
    text.setAttribute('y', y + 4);
    text.textContent = event.label;
    g.appendChild(text);

    // Session ID label (shortened) for global view
    if (!this.sessionId && event.session_id) {
      const sessionText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      sessionText.classList.add('session-label');
      sessionText.setAttribute('x', laneX - this.nodeRadius - 6);
      sessionText.setAttribute('y', y + 3);
      sessionText.setAttribute('text-anchor', 'end');
      sessionText.style.fontSize = '9px';
      sessionText.style.fill = sessionColor;
      sessionText.textContent = event.session_id.substring(0, 8);
      g.appendChild(sessionText);
    }

    // Time label
    const timeText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    timeText.classList.add('time-label');
    timeText.setAttribute('x', 4);
    timeText.setAttribute('y', y + 3);
    timeText.textContent = this.formatTime(event.ts);
    g.appendChild(timeText);

    // Click handler
    g.addEventListener('click', () => this.selectEvent(event));

    // Hover handler for tooltip
    g.addEventListener('mouseenter', (e) => this.showTooltip(e, event));
    g.addEventListener('mouseleave', () => this.hideTooltip());

    container.appendChild(g);
  }

  /**
   * Append a single new event node (incremental render)
   */
  appendEventNode(event) {
    const index = this.events.length - 1;
    const y = this.calculateY(index);

    // Debug: log laneWidth and calculated positions
    console.log('[DEBUG] appendEventNode:', {
      kind: event.kind,
      lane: event.lane,
      target_lane: event.target_lane,
      laneWidth: this.laneWidth,
      containerWidth: this.container.clientWidth,
      laneX: this.getLaneX(event.lane),
      targetX: event.target_lane ? this.getLaneX(event.target_lane) : null
    });

    // Extend SVG height if needed
    const currentHeight = parseInt(this.svg.getAttribute('height')) || 0;
    const requiredHeight = y + 50;
    if (requiredHeight > currentHeight) {
      this.svg.setAttribute('height', requiredHeight);
      this.drawLaneDividers(this.laneWidth || this.container.clientWidth, requiredHeight);
    }

    // Find or create events layer
    let eventsLayer = this.svg.querySelector('.events-layer');
    if (!eventsLayer) {
      eventsLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      eventsLayer.classList.add('events-layer');
      this.svg.appendChild(eventsLayer);
    }

    // Render the event
    this.renderEvent(event, y, eventsLayer, index);

    // Add 'new' class for animation
    const node = eventsLayer.querySelector(`[data-event-id="${event.id}"]`);
    if (node) {
      node.classList.add('new');
      setTimeout(() => node.classList.remove('new'), 500);
    }
  }

  /**
   * Select an event and show details
   */
  selectEvent(event) {
    this.selectedEventId = event.id;

    // Update visual selection
    this.svg.querySelectorAll('.event-node.selected').forEach(el => el.classList.remove('selected'));
    const node = this.svg.querySelector(`[data-event-id="${event.id}"]`);
    if (node) node.classList.add('selected');

    // Load detail
    this.loadDetail(event);
  }

  /**
   * Load event detail into panel
   */
  async loadDetail(event) {
    this.detailPanel.innerHTML = '<div class="loading">Loading...</div>';

    try {
      const data = await api(`/dumps/${event.dump_id}/raw`);

      this.detailPanel.innerHTML = `
        <div class="detail-header">
          <span class="badge ${this.kindClass(event.kind)}">${esc(event.kind)}</span>
          <span class="ts">${esc(this.formatDateTime(event.ts))}</span>
        </div>
        <div class="detail-header">
          <span class="message-id">session: ${esc(event.session_id || '-')}</span>
        </div>
        <div class="detail-header">
          <span class="message-id">message: ${esc(event.message_id || '-')}</span>
        </div>
        ${event.summary ? `<div class="detail-summary">${esc(event.summary)}</div>` : ''}
        <div class="detail-actions">
          <button onclick="SwimlaneTimeline.copyJson(this)">Copy JSON</button>
          <button onclick="SwimlaneTimeline.downloadJson(this, '${esc(event.kind)}')">Download</button>
          <button onclick="SwimlaneTimeline.expandAllJson(this)">Expand All</button>
          <button onclick="SwimlaneTimeline.collapseAllJson(this)">Collapse All</button>
        </div>
        <div class="json-viewer-wrap">
          <div class="json-viewer" aria-label="JSON detail viewer"></div>
        </div>
      `;
      SwimlaneTimeline.renderJsonViewer(this.detailPanel, data);
    } catch (err) {
      this.detailPanel.innerHTML = `<div class="error">Failed to load: ${err.message}</div>`;
    }
  }

  /**
   * Copy JSON to clipboard
   */
  static copyJson(button) {
    const panel = button.closest('.detail-panel');
    const raw = SwimlaneTimeline.getPanelJsonText(panel);
    if (!raw) return;
    navigator.clipboard.writeText(raw).then(() => {
      button.textContent = 'Copied!';
      setTimeout(() => button.textContent = 'Copy JSON', 1500);
    });
  }

  /**
   * Download JSON as file
   */
  static downloadJson(button, kind) {
    const panel = button.closest('.detail-panel');
    const raw = SwimlaneTimeline.getPanelJsonText(panel);
    if (!raw) return;
    const blob = new Blob([raw], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${kind}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /**
   * Expand all nodes in JSON viewer
   */
  static expandAllJson(button) {
    const panel = button.closest('.detail-panel');
    SwimlaneTimeline.setAllJsonNodesExpanded(panel, true);
  }

  /**
   * Collapse all nodes in JSON viewer
   */
  static collapseAllJson(button) {
    const panel = button.closest('.detail-panel');
    SwimlaneTimeline.setAllJsonNodesExpanded(panel, false);
  }

  /**
   * Get serialized JSON text from panel state
   */
  static getPanelJsonText(panel) {
    if (!panel) return '';
    if (typeof panel._rawJsonText === 'string' && panel._rawJsonText.length > 0) {
      return panel._rawJsonText;
    }
    return '';
  }

  /**
   * Render interactive JSON viewer with folding and path copy
   */
  static renderJsonViewer(panel, data) {
    if (!panel) return;
    const viewer = panel.querySelector('.json-viewer');
    if (!viewer) return;

    panel._rawJsonText = fmtJson(data);
    panel._jsonData = data;
    viewer.innerHTML = '';

    const lineCounter = { value: 1 };
    const rootPath = '$';
    const rootNode = SwimlaneTimeline.renderJsonNode({
      value: data,
      key: '',
      path: rootPath,
      depth: 0,
      isLast: true,
      lineCounter,
      defaultExpandDepth: 2
    });
    viewer.appendChild(rootNode);

    viewer.addEventListener('click', (e) => {
      const toggle = e.target.closest('.json-toggle');
      if (toggle) {
        e.preventDefault();
        const node = toggle.closest('.json-node');
        if (node) {
          const expanded = node.dataset.expanded === '1';
          SwimlaneTimeline.setJsonNodeExpanded(node, !expanded);
        }
        return;
      }

      const copyBtn = e.target.closest('.json-copy-path');
      if (copyBtn) {
        e.preventDefault();
        const path = copyBtn.getAttribute('data-path') || '';
        if (!path) return;
        navigator.clipboard.writeText(path).then(() => {
          const original = copyBtn.textContent;
          copyBtn.textContent = 'Copied';
          setTimeout(() => {
            copyBtn.textContent = original;
          }, 1000);
        });
      }
    });
  }

  /**
   * Render one JSON node recursively
   */
  static renderJsonNode(opts) {
    const { value, key, isArrayIndex, path, depth, isLast, lineCounter, defaultExpandDepth } = opts;
    const isArray = Array.isArray(value);
    const isObject = value && typeof value === 'object' && !isArray;
    const isContainer = isArray || isObject;

    if (!isContainer) {
      return SwimlaneTimeline.renderPrimitiveRow({ value, key, isArrayIndex, path, depth, isLast, lineCounter });
    }

    const node = document.createElement('div');
    node.className = 'json-node';
    const expanded = depth < defaultExpandDepth;
    node.dataset.expanded = expanded ? '1' : '0';

    const openRow = document.createElement('div');
    openRow.className = 'json-row json-open';
    openRow.style.setProperty('--depth', String(depth));
    openRow.appendChild(SwimlaneTimeline.lineNo(lineCounter.value++));

    const toggle = document.createElement('button');
    toggle.className = 'json-toggle';
    toggle.type = 'button';
    toggle.textContent = expanded ? '▾' : '▸';
    toggle.setAttribute('aria-label', expanded ? 'Collapse' : 'Expand');
    openRow.appendChild(toggle);

    if (key !== '') {
      if (isArrayIndex) {
        openRow.appendChild(SwimlaneTimeline.indexSpan(key));
      } else {
        openRow.appendChild(SwimlaneTimeline.keySpan(key));
      }
      openRow.appendChild(SwimlaneTimeline.literal(': '));
    }
    openRow.appendChild(SwimlaneTimeline.bracketSpan(isArray ? '[' : '{'));
    openRow.appendChild(SwimlaneTimeline.copyPathButton(path));
    node.appendChild(openRow);

    const childrenWrap = document.createElement('div');
    childrenWrap.className = 'json-children';
    if (!expanded) childrenWrap.style.display = 'none';
    node.appendChild(childrenWrap);

    const entries = isArray
      ? value.map((item, idx) => [idx, item])
      : Object.entries(value);
    entries.forEach(([childKey, childValue], idx) => {
      const childPath = SwimlaneTimeline.joinPath(path, childKey, isArray);
      childrenWrap.appendChild(SwimlaneTimeline.renderJsonNode({
        value: childValue,
        key: String(childKey),
        isArrayIndex: isArray,
        path: childPath,
        depth: depth + 1,
        isLast: idx === entries.length - 1,
        lineCounter,
        defaultExpandDepth
      }));
    });

    const closeRow = document.createElement('div');
    closeRow.className = 'json-row json-close';
    closeRow.style.setProperty('--depth', String(depth));
    closeRow.appendChild(SwimlaneTimeline.lineNo(lineCounter.value++));
    closeRow.appendChild(SwimlaneTimeline.literal(isArray ? ']' : '}'));
    if (!isLast) closeRow.appendChild(SwimlaneTimeline.literal(','));
    node.appendChild(closeRow);

    return node;
  }

  /**
   * Render primitive value row
   */
  static renderPrimitiveRow(opts) {
    const { value, key, isArrayIndex, path, depth, isLast, lineCounter } = opts;
    const row = document.createElement('div');
    row.className = 'json-row';
    row.style.setProperty('--depth', String(depth));
    row.appendChild(SwimlaneTimeline.lineNo(lineCounter.value++));
    if (key !== '') {
      if (isArrayIndex) {
        row.appendChild(SwimlaneTimeline.indexSpan(key));
      } else {
        row.appendChild(SwimlaneTimeline.keySpan(key));
      }
      row.appendChild(SwimlaneTimeline.literal(': '));
    }
    row.appendChild(SwimlaneTimeline.valueSpan(value));
    row.appendChild(SwimlaneTimeline.copyPathButton(path));
    if (!isLast) row.appendChild(SwimlaneTimeline.literal(','));
    return row;
  }

  static setAllJsonNodesExpanded(panel, expanded) {
    if (!panel) return;
    const nodes = panel.querySelectorAll('.json-node');
    nodes.forEach((node) => SwimlaneTimeline.setJsonNodeExpanded(node, expanded));
  }

  static setJsonNodeExpanded(node, expanded) {
    node.dataset.expanded = expanded ? '1' : '0';
    const toggle = node.querySelector(':scope > .json-row .json-toggle');
    if (toggle) {
      toggle.textContent = expanded ? '▾' : '▸';
      toggle.setAttribute('aria-label', expanded ? 'Collapse' : 'Expand');
    }
    const children = node.querySelector(':scope > .json-children');
    if (children) {
      children.style.display = expanded ? '' : 'none';
    }
  }

  static joinPath(parentPath, key, parentIsArray) {
    const keyText = String(key);
    if (parentIsArray) {
      return `${parentPath}[${keyText}]`;
    }
    if (/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(keyText)) {
      return parentPath === '$' ? `$.${keyText}` : `${parentPath}.${keyText}`;
    }
    return `${parentPath}[${JSON.stringify(keyText)}]`;
  }

  static lineNo(number) {
    const el = document.createElement('span');
    el.className = 'json-line-no';
    el.textContent = String(number);
    return el;
  }

  static literal(text) {
    const el = document.createElement('span');
    el.className = 'json-lit';
    el.textContent = text;
    return el;
  }

  static keySpan(key) {
    const el = document.createElement('span');
    el.className = 'json-key';
    el.textContent = `"${key}"`;
    return el;
  }

  static indexSpan(key) {
    const el = document.createElement('span');
    el.className = 'json-index';
    el.textContent = `[${key}]`;
    return el;
  }

  static bracketSpan(text) {
    const el = document.createElement('span');
    el.className = 'json-bracket';
    el.textContent = text;
    return el;
  }

  static copyPathButton(path) {
    const btn = document.createElement('button');
    btn.className = 'json-copy-path';
    btn.type = 'button';
    btn.setAttribute('data-path', path || '$');
    btn.textContent = 'Copy Path';
    return btn;
  }

  static valueSpan(value) {
    const el = document.createElement('span');
    const MAX_VALUE_CHARS = 120;

    if (value === null) {
      el.className = 'json-null';
      el.textContent = 'null';
      return el;
    }

    const valueType = typeof value;
    if (valueType === 'string') {
      el.className = 'json-string';
      const full = JSON.stringify(value);
      if (full.length > MAX_VALUE_CHARS) {
        el.textContent = `${full.slice(0, MAX_VALUE_CHARS)}...`;
        el.title = full;
      } else {
        el.textContent = full;
      }
      return el;
    }

    if (valueType === 'number') {
      el.className = 'json-number';
      el.textContent = String(value);
      return el;
    }

    if (valueType === 'boolean') {
      el.className = 'json-boolean';
      el.textContent = value ? 'true' : 'false';
      return el;
    }

    el.className = 'json-string';
    el.textContent = JSON.stringify(value);
    return el;
  }

  /**
   * Show tooltip on hover
   */
  showTooltip(e, event) {
    this.hideTooltip();

    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.id = 'eventTooltip';
    tooltip.innerHTML = `
      <strong>${esc(event.label)}</strong><br>
      <span style="color: #94a3b8;">${esc(event.summary || event.kind)}</span><br>
      <span style="color: #64748b; font-size: 11px;">session: ${esc(event.session_id || '-')}</span>
    `;

    document.body.appendChild(tooltip);

    // Position tooltip
    const rect = e.target.getBoundingClientRect();
    tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + 'px';
    tooltip.style.top = rect.bottom + 8 + 'px';
  }

  /**
   * Hide tooltip
   */
  hideTooltip() {
    const tooltip = document.getElementById('eventTooltip');
    if (tooltip) tooltip.remove();
  }

  /**
   * Show empty state
   */
  showEmpty() {
    this.container.innerHTML = `
      <div class="lane-headers">
        <div class="lane-header client">Anthropic Client</div>
        <div class="lane-header gateway">Gateway</div>
        <div class="lane-header oci">OCI GenAI</div>
      </div>
      <div class="empty-state">
        <h3>No Events Yet</h3>
        <p>Events will appear here as messages flow through the gateway.</p>
      </div>
    `;
  }

  /**
   * Show error message
   */
  showError(message) {
    this.container.innerHTML = `
      <div class="error" style="margin: 20px;">${esc(message)}</div>
    `;
  }

  /**
   * Show status message
   */
  showStatus(message) {
    const statusBar = document.getElementById('statusBar');
    if (statusBar) {
      statusBar.textContent = message;
      statusBar.classList.add('flash');
      setTimeout(() => statusBar.classList.remove('flash'), 800);
    }
  }

  /**
   * Scroll to bottom of container
   */
  scrollToBottom() {
    this.container.scrollTop = this.container.scrollHeight;
  }

  /**
   * Handle window resize
   */
  handleResize() {
    if (this.events.length > 0) {
      this.render();
    }
  }

  /**
   * Disconnect SSE and cleanup
   */
  disconnect() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    this.hideTooltip();
  }

  /**
   * Clear all debug data (frontend + backend)
   * Calls backend API to clear database and dump files,
   * then resets frontend state.
   */
  async clearAll() {
    // Call backend API to clear all data
    await api('/clear', { method: 'DELETE' });

    // Clear frontend state
    this.events = [];
    this.sessionColorMap = {};
    this.markerColorMap = {};
    this.selectedEventId = null;

    // Clear detail panel
    this.detailPanel.innerHTML = '<div class="placeholder">Click an event node to view details</div>';

    // Re-render empty state
    this.showEmpty();

    // Update event count
    this.updateEventCount();

    console.log('Cleared all debug data');
  }

  // Helper methods

  getLaneX(lane) {
    const width = this.laneWidth || this.container.clientWidth || 800;
    return width * (this.lanePositions[lane] || 0.5);
  }

  calculateY(index) {
    return this.headerHeight + 20 + index * this.nodeSpacingY;
  }

  sanitizeClass(kind) {
    return kind.replace(/[^a-zA-Z0-9_]/g, '_');
  }

  kindClass(kind) {
    if (kind.includes('request_summary')) return 'ok';
    if (kind.includes('tool_detection')) return 'warn';
    if (kind.includes('error')) return 'err';
    return '';
  }

  formatTime(isoString) {
    if (!isoString) return '';
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return isoString.substring(11, 19);
    }
  }

  formatDateTime(isoString) {
    if (!isoString) return '';
    try {
      const d = new Date(isoString);
      return d.toLocaleString('en-US', { hour12: false });
    } catch {
      return isoString;
    }
  }
}

// Export for use in HTML
window.SwimlaneTimeline = SwimlaneTimeline;
