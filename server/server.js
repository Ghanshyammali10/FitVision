require('dotenv').config();
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');

const app = express();
const server = http.createServer(app);

const PORT = process.env.PORT || 3001;
const CLIENT_URL = process.env.CLIENT_URL || '*';

app.use(cors({
  origin: CLIENT_URL === '*' ? true : CLIENT_URL,
  methods: ['GET', 'POST'],
  credentials: true
}));

const io = new Server(server, {
  cors: {
    origin: CLIENT_URL === '*' ? true : CLIENT_URL,
    methods: ['GET', 'POST'],
    credentials: true
  },
  pingInterval: 10000,
  pingTimeout: 5000
});

// In-memory rooms: { roomId: [{ socketId, role }] }
const rooms = {};

// Generate 6-character alphanumeric room ID
function generateRoomId() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let id = '';
  for (let i = 0; i < 6; i++) {
    id += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return id;
}

app.get('/', (req, res) => {
  res.json({
    status: 'FitVision Signaling Server is running',
    rooms: Object.keys(rooms).length,
    timestamp: new Date().toISOString()
  });
});

io.on('connection', (socket) => {
  console.log(`[Connect] ${socket.id}`);

  // Create a new room
  socket.on('create-room', (data, callback) => {
    let roomId = generateRoomId();
    while (rooms[roomId]) {
      roomId = generateRoomId();
    }
    rooms[roomId] = [];
    console.log(`[Room Created] ${roomId} by ${socket.id}`);
    if (typeof callback === 'function') {
      callback({ roomId });
    }
    socket.emit('room-created', { roomId });
  });

  // Join an existing room
  socket.on('join-room', ({ roomId, role, userId, userName }) => {
    if (!rooms[roomId]) {
      rooms[roomId] = [];
    }

    if (rooms[roomId].length >= 2) {
      socket.emit('room-full');
      console.log(`[Room Full] ${roomId} — rejected ${socket.id}`);
      return;
    }

    // Remove any previous room membership for this socket
    leaveAllRooms(socket);

    rooms[roomId].push({ socketId: socket.id, role, userId, userName });
    socket.join(roomId);
    socket.roomId = roomId;
    socket.userRole = role;

    console.log(`[Join Room] ${socket.id} as ${role} → room ${roomId} (${rooms[roomId].length}/2)`);

    // Notify other peers in the room
    socket.to(roomId).emit('user-joined', { role, userId, userName, socketId: socket.id });
  });

  // Relay WebRTC offer
  socket.on('offer', ({ roomId, offer }) => {
    console.log(`[Offer] from ${socket.id} in room ${roomId}`);
    socket.to(roomId).emit('offer', { offer, socketId: socket.id });
  });

  // Relay WebRTC answer
  socket.on('answer', ({ roomId, answer }) => {
    console.log(`[Answer] from ${socket.id} in room ${roomId}`);
    socket.to(roomId).emit('answer', { answer, socketId: socket.id });
  });

  // Relay ICE candidate
  socket.on('ice-candidate', ({ roomId, candidate }) => {
    socket.to(roomId).emit('ice-candidate', { candidate, socketId: socket.id });
  });

  // Relay garment captured event
  socket.on('garment-captured', (data) => {
    const { roomId } = data;
    console.log(`[Garment Captured] in room ${roomId}`);
    socket.to(roomId).emit('garment-captured', data);
  });

  // Leave room
  socket.on('leave-room', ({ roomId }) => {
    handleLeaveRoom(socket, roomId);
  });

  // Disconnect
  socket.on('disconnect', () => {
    console.log(`[Disconnect] ${socket.id}`);
    leaveAllRooms(socket);
  });
});

function handleLeaveRoom(socket, roomId) {
  if (!roomId || !rooms[roomId]) return;

  rooms[roomId] = rooms[roomId].filter(p => p.socketId !== socket.id);
  socket.leave(roomId);
  socket.to(roomId).emit('user-left', { socketId: socket.id });

  console.log(`[Leave Room] ${socket.id} left ${roomId} (${rooms[roomId].length} remaining)`);

  // Clean up empty rooms
  if (rooms[roomId].length === 0) {
    delete rooms[roomId];
    console.log(`[Room Deleted] ${roomId}`);
  }
}

function leaveAllRooms(socket) {
  for (const roomId in rooms) {
    const idx = rooms[roomId].findIndex(p => p.socketId === socket.id);
    if (idx !== -1) {
      handleLeaveRoom(socket, roomId);
    }
  }
}

server.listen(PORT, () => {
  console.log(`\n🚀 FitVision Signaling Server`);
  console.log(`   Port: ${PORT}`);
  console.log(`   Client URL: ${CLIENT_URL}`);
  console.log(`   Environment: ${process.env.NODE_ENV || 'development'}\n`);
});
