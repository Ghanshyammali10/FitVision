// ==========================================
// FitVision — WebRTC Peer Connection Module
// Native RTCPeerConnection, no third-party libs
// ==========================================

const ICE_SERVERS = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' }
  ]
};

let peerConnection = null;
let localStream = null;
let remoteStreamCallback = null;

/**
 * Initialize WebRTC with local stream and remote stream callback
 */
function initWebRTC(stream, onRemoteStream) {
  localStream = stream;
  remoteStreamCallback = onRemoteStream;

  peerConnection = new RTCPeerConnection(ICE_SERVERS);

  // Add local stream tracks to peer connection
  if (localStream) {
    localStream.getTracks().forEach(track => {
      peerConnection.addTrack(track, localStream);
    });
  }

  // Handle remote stream
  peerConnection.ontrack = (event) => {
    console.log('[WebRTC] Remote track received:', event.track.kind);
    if (remoteStreamCallback && event.streams[0]) {
      remoteStreamCallback(event.streams[0]);
    }
  };

  // Handle ICE candidates
  peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
      // This will be handled by the caller
      if (peerConnection._onIceCandidate) {
        peerConnection._onIceCandidate(event.candidate);
      }
    }
  };

  // Connection state monitoring
  peerConnection.onconnectionstatechange = () => {
    console.log('[WebRTC] Connection state:', peerConnection.connectionState);
    if (peerConnection._onConnectionStateChange) {
      peerConnection._onConnectionStateChange(peerConnection.connectionState);
    }
  };

  peerConnection.oniceconnectionstatechange = () => {
    console.log('[WebRTC] ICE connection state:', peerConnection.iceConnectionState);
  };

  return peerConnection;
}

/**
 * Set callback for ICE candidates (to send via socket)
 */
function onIceCandidateGenerated(callback) {
  if (peerConnection) {
    peerConnection._onIceCandidate = callback;
  }
}

/**
 * Set callback for connection state changes
 */
function onConnectionStateChange(callback) {
  if (peerConnection) {
    peerConnection._onConnectionStateChange = callback;
  }
}

/**
 * Create and return an SDP offer
 */
async function createOffer() {
  if (!peerConnection) throw new Error('PeerConnection not initialized');

  const offer = await peerConnection.createOffer({
    offerToReceiveAudio: true,
    offerToReceiveVideo: true
  });
  await peerConnection.setLocalDescription(offer);
  console.log('[WebRTC] Offer created');
  return offer;
}

/**
 * Handle an incoming SDP offer, create and return an answer
 */
async function handleOffer(offer) {
  if (!peerConnection) throw new Error('PeerConnection not initialized');

  await peerConnection.setRemoteDescription(new RTCSessionDescription(offer));
  const answer = await peerConnection.createAnswer();
  await peerConnection.setLocalDescription(answer);
  console.log('[WebRTC] Answer created');
  return answer;
}

/**
 * Handle an incoming SDP answer
 */
async function handleAnswer(answer) {
  if (!peerConnection) throw new Error('PeerConnection not initialized');

  await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
  console.log('[WebRTC] Remote description set from answer');
}

/**
 * Handle an incoming ICE candidate
 */
async function handleIceCandidate(candidate) {
  if (!peerConnection) return;

  try {
    await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
  } catch (error) {
    console.error('[WebRTC] Error adding ICE candidate:', error);
  }
}

/**
 * Replace a track (e.g., when switching cameras) without renegotiation
 */
async function replaceTrack(newTrack) {
  if (!peerConnection) return;

  const sender = peerConnection.getSenders().find(s => s.track && s.track.kind === newTrack.kind);
  if (sender) {
    await sender.replaceTrack(newTrack);
    console.log('[WebRTC] Track replaced:', newTrack.kind);
  }
}

/**
 * Close the peer connection and clean up
 */
function closeConnection() {
  if (peerConnection) {
    peerConnection.close();
    peerConnection = null;
    console.log('[WebRTC] Connection closed');
  }
}

/**
 * Get the current peer connection
 */
function getPeerConnection() {
  return peerConnection;
}

// Make available globally
window.FVWebRTC = {
  initWebRTC,
  onIceCandidateGenerated,
  onConnectionStateChange,
  createOffer,
  handleOffer,
  handleAnswer,
  handleIceCandidate,
  replaceTrack,
  closeConnection,
  getPeerConnection
};
