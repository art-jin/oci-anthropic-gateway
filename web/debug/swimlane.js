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
    this.autoScroll = true;
    this.selectedEventId = null;
    this.onEventCount = null;  // Callback for event count updates

    // Layout constants
    this.laneWidth = 0;
    this.nodeRadius = 8;
    this.nodeSpacingY = 60;
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
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

    // Arrow markers for each direction
    const directions = [
      { id: 'arrow-right', orient: 'auto' },
      { id: 'arrow-left', orient: 'auto-start-reverse' }
    ];

    directions.forEach(dir => {
      const arrowMarker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
      arrowMarker.setAttribute('id', dir.id);
      arrowMarker.setAttribute('markerWidth', '8');
      arrowMarker.setAttribute('markerHeight', '6');
      arrowMarker.setAttribute('refX', '7');
      arrowMarker.setAttribute('refY', '3');
      arrowMarker.setAttribute('orient', dir.orient);
      const arrowPath = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      arrowPath.setAttribute('points', '0 0, 8 3, 0 6');
      arrowPath.classList.add('arrow-marker');
      arrowMarker.appendChild(arrowPath);
      defs.appendChild(arrowMarker);
    });

    this.svg.appendChild(defs);
  }

  /**
   * Connect to SSE endpoint for real-time updates
   */
  connectSSE() {
    const url = this.sessionId
      ? `/debug/api/sessions/${encodeURIComponent(this.sessionId)}/events`
      : '/debug/api/events';

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

    // Add to events list
    this.events.push(event);
    this.updateEventCount();

    // Incrementally render new event
    this.appendEventNode(event);

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
    const sessionColor = this.getSessionColor(event.session_id);
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

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.classList.add('connector');

      // Determine direction for arrow
      const isRightward = targetX > laneX;
      line.setAttribute('x1', laneX);
      line.setAttribute('y1', y);
      line.setAttribute('x2', targetX);
      line.setAttribute('y2', y);  // Horizontal line (same y)
      line.setAttribute('marker-end', isRightward ? 'url(#arrow-right)' : 'url(#arrow-left)');
      line.style.stroke = sessionColor;
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
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.classList.add('node', this.sanitizeClass(event.kind));
      circle.setAttribute('cx', laneX);
      circle.setAttribute('cy', y);
      circle.setAttribute('r', this.nodeRadius);
      circle.style.fill = sessionColor;
      g.appendChild(circle);

      // Small circle at target position if has target_lane
      if (event.target_lane) {
        const targetX = this.getLaneX(event.target_lane);
        const targetCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        targetCircle.classList.add('node', 'target', this.sanitizeClass(event.kind));
        targetCircle.setAttribute('cx', targetX);
        targetCircle.setAttribute('cy', y);
        targetCircle.setAttribute('r', this.nodeRadius * 0.6);
        targetCircle.style.fill = sessionColor;
        targetCircle.style.opacity = '0.7';
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
        </div>
        <pre class="json">${esc(fmtJson(data))}</pre>
      `;
    } catch (err) {
      this.detailPanel.innerHTML = `<div class="error">Failed to load: ${err.message}</div>`;
    }
  }

  /**
   * Copy JSON to clipboard
   */
  static copyJson(button) {
    const pre = button.closest('.detail-panel').querySelector('pre.json');
    if (pre) {
      navigator.clipboard.writeText(pre.textContent).then(() => {
        button.textContent = 'Copied!';
        setTimeout(() => button.textContent = 'Copy JSON', 1500);
      });
    }
  }

  /**
   * Download JSON as file
   */
  static downloadJson(button, kind) {
    const pre = button.closest('.detail-panel').querySelector('pre.json');
    if (pre) {
      const blob = new Blob([pre.textContent], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${kind}_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
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
    this.hideTooltip();
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
