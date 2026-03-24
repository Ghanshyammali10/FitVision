// ==========================================
// FitVision — WebRTC Module
// Native RTCPeerConnection
// ==========================================

const ICE_SERVERS = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
  { urls: 'stun:stun2.l.google.com:19302' }
];

let peerConnection = null;
let onIceCandidateCb = null;
let onTrackCb = null;
let onConnectionStateChangeCb = null;

/**
 * Initialize a new WebRTC peer connection
 */
function initWebRTC(localStream, onRemoteStream) {
  closeConnection();

  peerConnection = new RTCPeerConnection({ iceServers: ICE_SERVERS });

  // Add local tracks
  if (localStream) {
    localStream.getTracks().forEach(track => {
      peerConnection.addTrack(track, localStream);
    });
  }

  // Handle remote stream
  peerConnection.ontrack = (event) => {
    console.log('[WebRTC] Remote track received');
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
    const state = peerConnection.connectionState;
    console.log('[WebRTC] Connection state:', state);
    if (onConnectionStateChangeCb) onConnectionStateChangeCb(state);
  };

  peerConnection.oniceconnectionstatechange = () => {
    console.log('[WebRTC] ICE state:', peerConnection.iceConnectionState);
  };

  return peerConnection;
}

/**
 * Create and return an SDP offer
 */
async function createOffer() {
  if (!peerConnection) return null;
  const offer = await peerConnection.createOffer({
    offerToReceiveAudio: true,
    offerToReceiveVideo: true
  });
  await peerConnection.setLocalDescription(offer);
  return offer;
}

/**
 * Handle incoming SDP offer and create answer
 */
async function handleOffer(offer) {
  if (!peerConnection) return null;
  await peerConnection.setRemoteDescription(new RTCSessionDescription(offer));
  const answer = await peerConnection.createAnswer();
  await peerConnection.setLocalDescription(answer);
  return answer;
}

/**
 * Handle incoming SDP answer
 */
async function handleAnswer(answer) {
  if (!peerConnection) return;
  await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
}

/**
 * Handle incoming ICE candidate
 */
async function handleIceCandidate(candidate) {
  if (!peerConnection) return;
  try {
    await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
  } catch (err) {
    console.error('[WebRTC] Error adding ICE candidate:', err);
  }
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
