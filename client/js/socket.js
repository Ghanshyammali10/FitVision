// ==========================================
// FitVision — Socket.io Client Module
// Socket.io v4.7.2 via CDN (loaded by HTML)
// ==========================================

// Server URL — change this to your Railway URL in production
const SERVER_URL = 'http://localhost:3001';

let socket = null;

/**
 * Connect to the signaling server
 */
function connectSocket(serverUrl) {
  const url = serverUrl || SERVER_URL;
  socket = io(url, {
    transports: ['websocket', 'polling'],
    withCredentials: false,
    reconnection: true,
    reconnectionAttempts: 5,
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
 * Create a new room — returns Promise<roomId>
 */
function createRoom() {
  return new Promise((resolve, reject) => {
    if (!socket || !socket.connected) {
      reject(new Error('Socket not connected'));
      return;
    }

    socket.emit('create-room', {}, (response) => {
      if (response && response.roomId) {
        resolve(response.roomId);
      }
    });

    // Also listen for room-created event as fallback
    socket.once('room-created', (data) => {
      resolve(data.roomId);
    });

    // Timeout after 10s
    setTimeout(() => reject(new Error('Room creation timeout')), 10000);
  });
}

/**
 * Join an existing room
 */
function joinRoom(roomId, role, userId, userName) {
  if (!socket) return;
  socket.emit('join-room', { roomId, role, userId, userName });
}

/**
 * Send WebRTC offer
 */
function sendOffer(roomId, offer) {
  if (!socket) return;
  socket.emit('offer', { roomId, offer });
}

/**
 * Send WebRTC answer
 */
function sendAnswer(roomId, answer) {
  if (!socket) return;
  socket.emit('answer', { roomId, answer });
}

/**
 * Send ICE candidate
 */
function sendIceCandidate(roomId, candidate) {
  if (!socket) return;
  socket.emit('ice-candidate', { roomId, candidate });
}

/**
 * Leave the current room
 */
function leaveRoom(roomId) {
  if (!socket) return;
  socket.emit('leave-room', { roomId });
}

/**
 * Emit garment-captured to buyer
 */
function emitGarmentCaptured(roomId, garmentDataUrl, garmentName, bgRemoved) {
  if (!socket) return;
  socket.emit('garment-captured', {
    roomId,
    garmentDataUrl,
    garmentName,
    bgRemoved,
    timestamp: Date.now()
  });
}

// ── Event Listeners ──

function onUserJoined(callback) {
  if (!socket) return;
  socket.on('user-joined', callback);
}

function onOffer(callback) {
  if (!socket) return;
  socket.on('offer', callback);
}

function onAnswer(callback) {
  if (!socket) return;
  socket.on('answer', callback);
}

function onIceCandidate(callback) {
  if (!socket) return;
  socket.on('ice-candidate', callback);
}

function onUserLeft(callback) {
  if (!socket) return;
  socket.on('user-left', callback);
}

function onRoomFull(callback) {
  if (!socket) return;
  socket.on('room-full', callback);
}

function onGarmentCaptured(callback) {
  if (!socket) return;
  socket.on('garment-captured', callback);
}

/**
 * Get the current socket instance
 */
function getSocket() {
  return socket;
}

// Make available globally (since HTML loads via script tag)
window.FVSocket = {
  connectSocket,
  createRoom,
  joinRoom,
  sendOffer,
  sendAnswer,
  sendIceCandidate,
  leaveRoom,
  emitGarmentCaptured,
  onUserJoined,
  onOffer,
  onAnswer,
  onIceCandidate,
  onUserLeft,
  onRoomFull,
  onGarmentCaptured,
  getSocket
};
