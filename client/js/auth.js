// ==========================================
// FitVision — Firebase Auth Module
// Firebase v9 modular SDK via CDN ESM imports
// ==========================================

import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js';
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, onAuthStateChanged, updateProfile, browserLocalPersistence, setPersistence } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js';
import { getFirestore, doc, setDoc, getDoc, serverTimestamp } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js';
import { getStorage, ref, uploadString, getDownloadURL } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-storage.js';

// ═══════════════════════════════════════════
// FIREBASE CONFIG — Production
// ═══════════════════════════════════════════
const firebaseConfig = {
  apiKey: "AIzaSyAK5FBPXteGJzL0p_iZZ3T7diU3uTDwDg4",
  authDomain: "myapp-79553.firebaseapp.com",
  projectId: "myapp-79553",
  storageBucket: "myapp-79553.firebasestorage.app",
  messagingSenderId: "65124833552",
  appId: "1:65124833552:web:fb55f926e613a1c3496d95"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
const storage = getStorage(app);

// Set persistence to local
setPersistence(auth, browserLocalPersistence).catch(console.error);

/**
 * Register a new user with email/password and store role in Firestore
 */
async function registerUser(email, password, displayName, role) {
  try {
    const userCred = await createUserWithEmailAndPassword(auth, email, password);
    const user = userCred.user;
    await updateProfile(user, { displayName });
    await setDoc(doc(db, 'users', user.uid), {
      role, displayName, email,
      createdAt: serverTimestamp()
    });
    return { success: true, user, role };
  } catch (error) {
    let message = 'Registration failed';
    switch (error.code) {
      case 'auth/email-already-in-use': message = 'This email is already registered'; break;
      case 'auth/invalid-email': message = 'Invalid email address'; break;
      case 'auth/weak-password': message = 'Password must be at least 6 characters'; break;
      default: message = error.message;
    }
    return { success: false, error: message };
  }
}

/**
 * Login user and return role
 */
async function loginUser(email, password) {
  try {
    const userCred = await signInWithEmailAndPassword(auth, email, password);
    const user = userCred.user;
    const userDoc = await getDoc(doc(db, 'users', user.uid));
    if (!userDoc.exists()) {
      return { success: false, error: 'User data not found. Please register again.' };
    }
    const role = userDoc.data().role;
    return { success: true, user, role };
  } catch (error) {
    let message = 'Login failed';
    switch (error.code) {
      case 'auth/user-not-found': message = 'No account found with this email'; break;
      case 'auth/wrong-password':
      case 'auth/invalid-credential': message = 'Incorrect password'; break;
      case 'auth/invalid-email': message = 'Invalid email address'; break;
      case 'auth/too-many-requests': message = 'Too many attempts. Try again later'; break;
      default: message = error.message;
    }
    return { success: false, error: message };
  }
}

/**
 * Logout user and redirect to login page
 */
async function logoutUser() {
  try {
    await signOut(auth);
    window.location.href = 'index.html';
  } catch (error) {
    console.error('Logout error:', error);
  }
}

/**
 * Check auth state and protect routes
 */
function checkAuth(requiredRole) {
  return new Promise((resolve, reject) => {
    onAuthStateChanged(auth, async (user) => {
      if (!user) {
        window.location.href = 'index.html';
        reject(new Error('Not authenticated'));
        return;
      }
      try {
        const userDoc = await getDoc(doc(db, 'users', user.uid));
        if (!userDoc.exists()) {
          await signOut(auth);
          window.location.href = 'index.html';
          reject(new Error('User doc not found'));
          return;
        }
        const data = userDoc.data();
        const role = data.role;
        if (requiredRole && role !== requiredRole) {
          window.location.href = role === 'seller' ? 'seller.html' : 'buyer.html';
          reject(new Error('Role mismatch'));
          return;
        }
        resolve({
          user, uid: user.uid, role,
          displayName: data.displayName || user.displayName || 'User',
          email: data.email
        });
      } catch (error) {
        console.error('Auth check error:', error);
        reject(error);
      }
    });
  });
}

function getCurrentUser() {
  return auth.currentUser;
}

/**
 * Upload garment image to Firebase Storage
 */
async function uploadGarmentImage(sellerId, dataUrl) {
  try {
    const timestamp = Date.now();
    const storageRef = ref(storage, `garments/${sellerId}/${timestamp}.png`);
    const snapshot = await uploadString(storageRef, dataUrl, 'data_url');
    const downloadURL = await getDownloadURL(snapshot.ref);
    return { success: true, url: downloadURL };
  } catch (error) {
    console.error('Upload error:', error);
    return { success: false, error: error.message };
  }
}

/**
 * Save garment data to Firestore
 */
async function saveGarment(sellerId, roomId, imageUrl, name, bgRemoved) {
  try {
    const garmentRef = doc(db, 'garments', `${sellerId}_${Date.now()}`);
    await setDoc(garmentRef, {
      sellerId, roomId, imageUrl, name,
      capturedAt: serverTimestamp(),
      bgRemoved: bgRemoved || false
    });
    return { success: true };
  } catch (error) {
    console.error('Save garment error:', error);
    return { success: false, error: error.message };
  }
}

/**
 * Create room document in Firestore
 */
async function createRoomDoc(roomId, sellerId, sellerName) {
  try {
    await setDoc(doc(db, 'rooms', roomId), {
      sellerId, sellerName, status: 'waiting',
      createdAt: serverTimestamp()
    });
    return { success: true };
  } catch (error) {
    console.error('Create room doc error:', error);
    return { success: false };
  }
}

/**
 * Update room status
 */
async function updateRoomStatus(roomId, status) {
  try {
    await setDoc(doc(db, 'rooms', roomId), { status }, { merge: true });
  } catch (error) {
    console.error('Update room status error:', error);
  }
}

export {
  auth, db, storage,
  registerUser, loginUser, logoutUser,
  checkAuth, getCurrentUser,
  uploadGarmentImage, saveGarment,
  createRoomDoc, updateRoomStatus
};
