// ==========================================================================
// GestureVision AI — Client Application Logic
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const cameraSelect = document.getElementById('cameraSelect');
    const refreshCamerasBtn = document.getElementById('refreshCamerasBtn');
    const refreshIcon = document.getElementById('refreshIcon');
    const hudCameraLabel = document.getElementById('hudCameraLabel');
    const hudFps = document.getElementById('hudFps');
    const hudLatency = document.getElementById('hudLatency');
    const videoStream = document.getElementById('videoStream');
    const videoContainer = document.getElementById('videoContainer');

    const confidenceSlider = document.getElementById('confidenceSlider');
    const confidenceValue = document.getElementById('confidenceValue');
    const presetBtns = document.querySelectorAll('.preset-btn');

    const mirrorToggleBtn = document.getElementById('mirrorToggleBtn');
    const snapshotHeaderBtn = document.getElementById('snapshotHeaderBtn');
    const floatingSnapshotBtn = document.getElementById('floatingSnapshotBtn');
    const fullscreenBtn = document.getElementById('fullscreenBtn');

    // Gesture UI Tiles
    const tileOpenPalm = document.getElementById('tileOpenPalm');
    const statusOpenPalm = document.getElementById('statusOpenPalm');
    const meterOpenPalm = document.getElementById('meterOpenPalm');

    const tileThumbsUp = document.getElementById('tileThumbsUp');
    const statusThumbsUp = document.getElementById('statusThumbsUp');
    const meterThumbsUp = document.getElementById('meterThumbsUp');

    const activeCountBadge = document.getElementById('activeCountBadge');

    // Snapshots Elements
    const snapshotsCarousel = document.getElementById('snapshotsCarousel');
    const snapshotsCount = document.getElementById('snapshotsCount');
    const emptySnapshotsMsg = document.getElementById('emptySnapshotsMsg');

    // Modal Elements
    const imageModal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const modalTimestamp = document.getElementById('modalTimestamp');
    const modalDownloadBtn = document.getElementById('modalDownloadBtn');
    const closeModalBtn = document.getElementById('closeModalBtn');

    let debounceConfidenceTimer = null;

    // ─── Toast Notifications ──────────────────────────────────────────────
    function showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = 'toast';
        if (type === 'success') toast.style.borderLeftColor = '#10b981';
        if (type === 'error') toast.style.borderLeftColor = '#f43f5e';
        toast.innerHTML = `<span>${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.transition = 'opacity 0.3s, transform 0.3s';
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(50px)';
            setTimeout(() => toast.remove(), 300);
        }, 2600);
    }

    // ─── Camera Switching ─────────────────────────────────────────────────
    async function switchCamera(cameraIndex) {
        showToast(`Switching to Camera ${cameraIndex}...`);
        try {
            const res = await fetch('/api/set_camera', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ camera_index: cameraIndex })
            });
            const data = await res.json();
            if (data.success) {
                hudCameraLabel.textContent = `Camera ${data.current_camera}`;
                // Flush stream element
                videoStream.src = `/video_feed?t=${Date.now()}`;
                showToast(`Connected to Camera ${data.current_camera}`, 'success');
            } else {
                showToast(`Failed to open Camera ${cameraIndex}`, 'error');
            }
        } catch (err) {
            console.error('Error setting camera:', err);
            showToast('Error communicating with camera server', 'error');
        }
    }

    cameraSelect.addEventListener('change', (e) => {
        switchCamera(parseInt(e.target.value, 10));
    });

    // ─── Refresh Camera List ──────────────────────────────────────────────
    refreshCamerasBtn.addEventListener('click', async () => {
        refreshIcon.classList.add('spinning');
        showToast('Rescanning video devices...');
        try {
            const res = await fetch('/api/cameras?refresh=true');
            const data = await res.json();
            if (data.success) {
                cameraSelect.innerHTML = '';
                data.cameras.forEach(cam => {
                    const opt = document.createElement('option');
                    opt.value = cam;
                    opt.textContent = `Camera Index ${cam} ${cam === 0 ? '(Default)' : ''}`;
                    if (cam === data.current_camera) opt.selected = true;
                    cameraSelect.appendChild(opt);
                });
                hudCameraLabel.textContent = `Camera ${data.current_camera}`;
                showToast(`Discovered ${data.cameras.length} camera(s)`, 'success');
            }
        } catch (err) {
            console.error('Error refreshing cameras:', err);
            showToast('Failed to scan cameras', 'error');
        } finally {
            refreshIcon.classList.remove('spinning');
        }
    });

    // ─── Confidence Slider ────────────────────────────────────────────────
    function updateConfidence(val) {
        confidenceValue.textContent = `${val}%`;
        clearTimeout(debounceConfidenceTimer);
        debounceConfidenceTimer = setTimeout(async () => {
            try {
                await fetch('/api/set_confidence', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ confidence: val / 100 })
                });
            } catch (err) {
                console.error('Error updating confidence:', err);
            }
        }, 150);
    }

    confidenceSlider.addEventListener('input', (e) => {
        updateConfidence(parseInt(e.target.value, 10));
    });

    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const val = parseInt(btn.dataset.val, 10);
            confidenceSlider.value = val;
            updateConfidence(val);
        });
    });

    // ─── Mirror Toggle ────────────────────────────────────────────────────
    mirrorToggleBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/toggle_mirror', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                showToast(data.mirror ? 'Mirror Mode ON' : 'Mirror Mode OFF');
            }
        } catch (err) {
            console.error('Error toggling mirror:', err);
        }
    });

    // ─── Snapshot Capture ─────────────────────────────────────────────────
    async function takeSnapshot() {
        showToast('Capturing snapshot...');
        try {
            const res = await fetch('/api/snapshot', { method: 'POST' });
            const data = await res.json();
            if (data.success && data.snapshot) {
                showToast('Snapshot saved!', 'success');
                addSnapshotToGallery(data.snapshot, true);
            } else {
                showToast('Failed to capture snapshot', 'error');
            }
        } catch (err) {
            console.error('Error taking snapshot:', err);
            showToast('Snapshot capture failed', 'error');
        }
    }

    snapshotHeaderBtn.addEventListener('click', takeSnapshot);
    floatingSnapshotBtn.addEventListener('click', takeSnapshot);

    // ─── Fullscreen Toggle ────────────────────────────────────────────────
    fullscreenBtn.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            videoContainer.requestFullscreen().catch(err => {
                console.error(`Error attempting fullscreen: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    });

    // ─── Snapshots Gallery ────────────────────────────────────────────────
    function addSnapshotToGallery(item, prepend = false) {
        if (emptySnapshotsMsg) {
            emptySnapshotsMsg.style.display = 'none';
        }

        const thumb = document.createElement('div');
        thumb.className = 'snapshot-thumb';
        thumb.title = `Captured at ${item.timestamp}`;
        thumb.innerHTML = `<img src="${item.url}" alt="${item.filename}" />`;

        thumb.addEventListener('click', () => {
            modalImage.src = item.url;
            modalTimestamp.textContent = `Timestamp: ${item.timestamp} | Camera: ${item.camera ?? '-'}`;
            modalDownloadBtn.href = item.url;
            modalDownloadBtn.download = item.filename;
            imageModal.classList.add('open');
        });

        if (prepend) {
            snapshotsCarousel.prepend(thumb);
        } else {
            snapshotsCarousel.appendChild(thumb);
        }

        const total = snapshotsCarousel.querySelectorAll('.snapshot-thumb').length;
        snapshotsCount.textContent = `${total} saved`;
    }

    async function loadScreenshots() {
        try {
            const res = await fetch('/api/screenshots');
            const data = await res.json();
            if (data.success && data.screenshots.length > 0) {
                if (emptySnapshotsMsg) emptySnapshotsMsg.style.display = 'none';
                data.screenshots.forEach(item => addSnapshotToGallery(item, false));
            }
        } catch (err) {
            console.error('Error loading screenshots:', err);
        }
    }
    loadScreenshots();

    // Modal Close
    closeModalBtn.addEventListener('click', () => imageModal.classList.remove('open'));
    imageModal.addEventListener('click', (e) => {
        if (e.target === imageModal) imageModal.classList.remove('open');
    });

    // ─── Real-Time Stats & Gesture Polling (every 250ms) ───────────────────
    async function pollStats() {
        try {
            const res = await fetch('/api/stats');
            if (!res.ok) return;
            const data = await res.json();

            // Update Telemetry HUD
            hudFps.textContent = data.fps.toFixed(1);
            hudLatency.textContent = data.inference_ms.toFixed(1);

            const detections = data.detections || {};
            let totalDetections = 0;

            // 1. Open Palm
            if (detections['Open Palm']) {
                const confPct = Math.round(detections['Open Palm'].max_conf * 100);
                tileOpenPalm.classList.add('active-palm');
                statusOpenPalm.textContent = `Detected (${confPct}%)`;
                meterOpenPalm.style.width = `${confPct}%`;
                totalDetections += detections['Open Palm'].count;
            } else {
                tileOpenPalm.classList.remove('active-palm');
                statusOpenPalm.textContent = 'Not Detected';
                meterOpenPalm.style.width = '0%';
            }

            // 2. Thumbs Up
            if (detections['Thumbs Up']) {
                const confPct = Math.round(detections['Thumbs Up'].max_conf * 100);
                tileThumbsUp.classList.add('active-thumb');
                statusThumbsUp.textContent = `Detected (${confPct}%)`;
                meterThumbsUp.style.width = `${confPct}%`;
                totalDetections += detections['Thumbs Up'].count;
            } else {
                tileThumbsUp.classList.remove('active-thumb');
                statusThumbsUp.textContent = 'Not Detected';
                meterThumbsUp.style.width = '0%';
            }

            activeCountBadge.textContent = `${totalDetections} Detected`;

        } catch (err) {
            // Server might be busy; silently retry on next cycle
        }
    }

    setInterval(pollStats, 250);

    // ─── Global Keyboard Shortcuts ────────────────────────────────────────
    window.addEventListener('keydown', (e) => {
        // Prevent shortcuts if typing in input
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

        if (e.code === 'Space') {
            e.preventDefault();
            takeSnapshot();
        } else if (e.key === 'm' || e.key === 'M') {
            mirrorToggleBtn.click();
        } else if (e.key === 'c' || e.key === 'C') {
            // Cycle camera
            const options = Array.from(cameraSelect.options);
            if (options.length > 1) {
                const currIdx = cameraSelect.selectedIndex;
                const nextIdx = (currIdx + 1) % options.length;
                cameraSelect.selectedIndex = nextIdx;
                switchCamera(parseInt(options[nextIdx].value, 10));
            }
        }
    });
});
