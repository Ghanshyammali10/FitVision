// ==========================================
// FitVision — Socket.io Client Module
// ==========================================

// Production signaling server URL
const SERVER_URL = 'https://fitvision-production.up.railway.app';

let socket = null;

/**
 * Connect to the signaling server
 */
function connectSocket(serverUrl) {
  const url = serverUrl || SERVER_URL;
  socket = io(url, {
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionAttempts: 10,
    reconnectionDelay: 2000
  });

  socket.on('connect', () => {
    console.log('[Socket] Connected:', socket.id);
  });

  socket.on('disconnect', (reason) => {
    console.log('[Socket] Disconnected:', reason);
  });

  socket.on('connect_error', (err) => {
    console.error('[Socket] Connection error:', err.message);
  });

  return socket;
}

/**
 * Get the current socket instance
 */
function getSocket() {
  return socket;
}

// ── Emitters ──

function createRoom(callback) {
  if (!socket) return;
  socket.emit('create-room', {}, callback);
}

function joinRoom(roomId, role, userId, userName) {
  if (!socket) return;
  socket.emit('join-room', { roomId, role, userId, userName });
}

function leaveRoom(roomId) {
  if (!socket) return;
  socket.emit('leave-room', { roomId });
}

function sendOffer(roomId, offer) {
  if (!socket) return;
  socket.emit('offer', { roomId, offer });
}

function sendAnswer(roomId, answer) {
  if (!socket) return;
  socket.emit('answer', { roomId, answer });
}

function sendIceCandidate(roomId, candidate) {
  if (!socket) return;
  socket.emit('ice-candidate', { roomId, candidate });
}

function sendGarmentCaptured(roomId, garmentDataUrl, garmentName) {
  if (!socket) return;
  socket.emit('garment-captured', {
    roomId, garmentDataUrl, garmentName,
    timestamp: Date.now()
  });
}

// ── Listeners ──

function onRoomCreated(cb)     { if (socket) socket.on('room-created', cb); }
function onUserJoined(cb)      { if (socket) socket.on('user-joined', cb); }
function onUserLeft(cb)        { if (socket) socket.on('user-left', cb); }
function onRoomFull(cb)        { if (socket) socket.on('room-full', cb); }
function onOffer(cb)           { if (socket) socket.on('offer', cb); }
function onAnswer(cb)          { if (socket) socket.on('answer', cb); }
function onIceCandidate(cb)    { if (socket) socket.on('ice-candidate', cb); }
function onGarmentCaptured(cb) { if (socket) socket.on('garment-captured', cb); }
function onReplaced(cb)        { if (socket) socket.on('replaced', cb); }

// Expose globally
window.FVSocket = {
  connectSocket, getSocket,
  createRoom, joinRoom, leaveRoom,
  sendOffer, sendAnswer, sendIceCandidate,
  sendGarmentCaptured,
  onRoomCreated, onUserJoined, onUserLeft, onRoomFull,
  onOffer, onAnswer, onIceCandidate,
  onGarmentCaptured, onReplaced
};
