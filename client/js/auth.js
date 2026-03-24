// ==========================================
// FitVision — Firebase Auth Module
// Firebase v9 modular SDK via CDN ESM imports
// ==========================================

import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js';
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, onAuthStateChanged, updateProfile, browserLocalPersistence, setPersistence } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js';
import { getFirestore, doc, setDoc, getDoc, serverTimestamp } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js';
import { getStorage, ref, uploadString, getDownloadURL } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-storage.js';

// ═══════════════════════════════════════════
// PASTE YOUR FIREBASE CONFIG HERE
// Get it from Firebase Console → Project Settings → General → Your apps → Web app
// ═══════════════════════════════════════════
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
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

    // Update display name
    await updateProfile(user, { displayName });

    // Store user data in Firestore
    await setDoc(doc(db, 'users', user.uid), {
      role,
      displayName,
      email,
      createdAt: serverTimestamp()
    });

    return { success: true, user, role };
  } catch (error) {
    let message = 'Registration failed';
    switch (error.code) {
      case 'auth/email-already-in-use':
        message = 'This email is already registered';
        break;
      case 'auth/invalid-email':
        message = 'Invalid email address';
        break;
      case 'auth/weak-password':
        message = 'Password must be at least 6 characters';
        break;
      default:
        message = error.message;
    }
    return { success: false, error: message };
  }
}

/**
 * Login user and redirect based on role
 */
async function loginUser(email, password) {
  try {
    const userCred = await signInWithEmailAndPassword(auth, email, password);
    const user = userCred.user;

    // Fetch role from Firestore
    const userDoc = await getDoc(doc(db, 'users', user.uid));
    if (!userDoc.exists()) {
      return { success: false, error: 'User data not found. Please register again.' };
    }

    const role = userDoc.data().role;
    return { success: true, user, role };
  } catch (error) {
    let message = 'Login failed';
    switch (error.code) {
      case 'auth/user-not-found':
        message = 'No account found with this email';
        break;
      case 'auth/wrong-password':
      case 'auth/invalid-credential':
        message = 'Incorrect password';
        break;
      case 'auth/invalid-email':
        message = 'Invalid email address';
        break;
      case 'auth/too-many-requests':
        message = 'Too many attempts. Please try again later';
        break;
      default:
        message = error.message;
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
 * @param {string} requiredRole - 'seller' or 'buyer' — the role this page requires
 * @returns {Promise<{user, role, displayName}>}
 */
function checkAuth(requiredRole) {
  return new Promise((resolve, reject) => {
    onAuthStateChanged(auth, async (user) => {
      if (!user) {
        // Not logged in → redirect to login
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

        // Role mismatch → redirect to correct page
        if (requiredRole && role !== requiredRole) {
          window.location.href = role === 'seller' ? 'seller.html' : 'buyer.html';
          reject(new Error('Role mismatch'));
          return;
        }

        resolve({
          user,
          uid: user.uid,
          role,
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

/**
 * Get current user synchronously (returns null if not yet loaded)
 */
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
      sellerId,
      roomId,
      imageUrl,
      name,
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
      sellerId,
      sellerName,
      status: 'waiting',
      createdAt: serverTimestamp()
    });
    return { success: true };
  } catch (error) {
    console.error('Create room doc error:', error);
    return { success: false };
  }
}

/**
 * Update room status in Firestore
 */
async function updateRoomStatus(roomId, status) {
  try {
    await setDoc(doc(db, 'rooms', roomId), { status }, { merge: true });
  } catch (error) {
    console.error('Update room status error:', error);
  }
}

// Export everything
export {
  auth, db, storage,
  registerUser,
  loginUser,
  logoutUser,
  checkAuth,
  getCurrentUser,
  uploadGarmentImage,
  saveGarment,
  createRoomDoc,
  updateRoomStatus
};
