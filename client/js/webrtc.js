// ==========================================
// FitVision — WebRTC Module (v2 — Resilient)
// Native RTCPeerConnection + TURN + ICE Buffering
// ==========================================

const ICE_SERVERS = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
  // TURN relay — critical for NATs that block direct P2P
  {
    urls: 'turn:a.relay.metered.ca:80',
    username: 'e8dd65b92a0d8b14a6691060',
    credential: '2D9tFpPYqEi4q6Wt'
  },
  {
    urls: 'turn:a.relay.metered.ca:443?transport=tcp',
    username: 'e8dd65b92a0d8b14a6691060',
    credential: '2D9tFpPYqEi4q6Wt'
  }
];

let peerConnection = null;
let onIceCandidateCb = null;
let onTrackCb = null;
let onConnectionStateChangeCb = null;
let pendingCandidates = []; // Buffer ICE candidates until remote description is set

/**
 * Initialize a new WebRTC peer connection
 */
function initWebRTC(localStream, onRemoteStream) {
  closeConnection();
  pendingCandidates = [];

  peerConnection = new RTCPeerConnection({ iceServers: ICE_SERVERS });

  // Add local tracks
  if (localStream) {
    localStream.getTracks().forEach(track => {
      peerConnection.addTrack(track, localStream);
    });
  }

  // Handle remote stream
  peerConnection.ontrack = (event) => {
    console.log('[WebRTC] Remote track received:', event.track.kind);
    if (onRemoteStream && event.streams[0]) {
      onRemoteStream(event.streams[0]);
    }
    if (onTrackCb) onTrackCb(event);
  };

  // ICE candidates
  peerConnection.onicecandidate = (event) => {
    if (event.candidate && onIceCandidateCb) {
      onIceCandidateCb(event.candidate);
    }
  };

  // Connection state monitoring
  peerConnection.onconnectionstatechange = () => {
    const state = peerConnection ? peerConnection.connectionState : 'closed';
    console.log('[WebRTC] Connection state:', state);
    if (onConnectionStateChangeCb) onConnectionStateChangeCb(state);
  };

  peerConnection.oniceconnectionstatechange = () => {
    if (!peerConnection) return;
    const iceState = peerConnection.iceConnectionState;
    console.log('[WebRTC] ICE state:', iceState);
    // Auto-recovery: if ICE fails, try restarting
    if (iceState === 'failed') {
      console.warn('[WebRTC] ICE failed — attempting restart');
      try { peerConnection.restartIce(); } catch (e) { console.error('[WebRTC] ICE restart failed:', e); }
    }
  };

  return peerConnection;
}

/**
 * Create and return an SDP offer
 */
async function createOffer() {
  if (!peerConnection) return null;
  try {
    const offer = await peerConnection.createOffer({
      offerToReceiveAudio: true,
      offerToReceiveVideo: true
    });
    await peerConnection.setLocalDescription(offer);
    return offer;
  } catch (err) {
    console.error('[WebRTC] Error creating offer:', err);
    return null;
  }
}

/**
 * Handle incoming SDP offer and create answer
 */
async function handleOffer(offer) {
  if (!peerConnection) return null;
  try {
    await peerConnection.setRemoteDescription(new RTCSessionDescription(offer));
    // Flush any ICE candidates that arrived before remote description
    await flushPendingCandidates();
    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);
    return answer;
  } catch (err) {
    console.error('[WebRTC] Error handling offer:', err);
    return null;
  }
}

/**
 * Handle incoming SDP answer
 */
async function handleAnswer(answer) {
  if (!peerConnection) return;
  try {
    await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
    // Flush any ICE candidates that arrived before remote description
    await flushPendingCandidates();
  } catch (err) {
    console.error('[WebRTC] Error handling answer:', err);
  }
}

/**
 * Handle incoming ICE candidate — buffers if remote description isn't set yet
 */
async function handleIceCandidate(candidate) {
  if (!peerConnection) return;

  // If remote description isn't set yet, buffer the candidate
  if (!peerConnection.remoteDescription || !peerConnection.remoteDescription.type) {
    pendingCandidates.push(candidate);
    return;
  }

  try {
    await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
  } catch (err) {
    console.error('[WebRTC] Error adding ICE candidate:', err);
  }
}

/**
 * Flush buffered ICE candidates after remote description is set
 */
async function flushPendingCandidates() {
  if (!peerConnection || pendingCandidates.length === 0) return;
  console.log(`[WebRTC] Flushing ${pendingCandidates.length} buffered ICE candidates`);
  for (const candidate of pendingCandidates) {
    try {
      await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
    } catch (err) {
      console.error('[WebRTC] Error adding buffered ICE candidate:', err);
    }
  }
  pendingCandidates = [];
}

/**
 * Replace video track (e.g. camera switch)
 */
function replaceVideoTrack(newTrack) {
  if (!peerConnection) return;
  const sender = peerConnection.getSenders().find(s => s.track && s.track.kind === 'video');
  if (sender) {
    sender.replaceTrack(newTrack);
  }
}

/**
 * Close the peer connection
 */
function closeConnection() {
  if (peerConnection) {
    peerConnection.close();
    peerConnection = null;
  }
  pendingCandidates = [];
}

function onIceCandidateGenerated(cb) { onIceCandidateCb = cb; }
function onRemoteTrack(cb) { onTrackCb = cb; }
function onConnectionState(cb) { onConnectionStateChangeCb = cb; }

function getPeerConnection() { return peerConnection; }

// Expose globally
window.FVWebRTC = {
  initWebRTC, createOffer, handleOffer, handleAnswer,
  handleIceCandidate, replaceVideoTrack, closeConnection,
  onIceCandidateGenerated, onRemoteTrack, onConnectionState,
  getPeerConnection
};
