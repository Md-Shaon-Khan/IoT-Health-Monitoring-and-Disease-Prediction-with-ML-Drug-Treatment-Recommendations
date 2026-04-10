/**
 * HealthBridge | Auth Logic V4.1
 * Corrected for Nordic Clinical Theme and Dual-Contact Logic
 */

const API_BASE = window.location.origin;

let currentRole = 'patient';

function setRole(role) {
    currentRole = role;
    document.querySelectorAll('.role-option').forEach(opt => opt.classList.remove('active'));
    
    const selectedRole = document.querySelector(`[data-role="${role}"]`);
    if (selectedRole) selectedRole.classList.add('active');
    
    // Toggle Department field visibility for doctors
    const deptField = document.getElementById('deptField');
    if (deptField) {
        // Only show if role is doctor AND we are in signup mode
        const isSignup = document.getElementById('submitBtn').innerText === "Sign Up";
        deptField.classList.toggle('hidden', role !== 'doctor' || !isSignup);
    }
}

function showMode(mode) {
    const isSignup = mode === 'signup';
    
    // UI Classes
    document.getElementById('toggleSignin').classList.toggle('active', !isSignup);
    document.getElementById('toggleSignup').classList.toggle('active', isSignup);
    
    // Text Updates
    document.getElementById('formTitle').innerText = isSignup ? "Create Account" : "Access Portal";
    document.getElementById('submitBtn').innerText = isSignup ? "Sign Up" : "Sign In";
    
    // Section Visibility
    const toggleFields = ['roleSelection', 'nameField', 'contactSection', 'bloodGroupField'];
    toggleFields.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.toggle('hidden', !isSignup);
    });

    // Special handling for Department Field
    const deptField = document.getElementById('deptField');
    if (deptField) {
        deptField.classList.toggle('hidden', !isSignup || currentRole !== 'doctor');
    }
}

document.getElementById('authForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const mode = document.getElementById('submitBtn').innerText;
    
    const emailInput = document.getElementById('userEmail');
    const phoneInput = document.getElementById('userPhone');
    
    const email = emailInput ? emailInput.value.trim() : "";
    const phone = phoneInput ? phoneInput.value.trim() : "";

    // Validation: Email OR Phone required for Sign Up
    if (mode === "Sign Up" && !email && !phone) {
        alert("Institutional records require either an Email Address or a Phone Number.");
        return;
    }

    const data = {
        user_id: document.getElementById('userId').value.trim(),
        name: document.getElementById('userName').value.trim() || "User",
        email: email || null,
        phone: phone || null,
        role: currentRole,
        password: "password123", // Static password as per your design
        dept: currentRole === 'doctor' ? document.getElementById('department').value.trim() : null,
        blood_group: document.getElementById('bloodGroup').value || null
    };

    const endpoint = mode.trim().toLowerCase() === "sign up" ? "signup" : "login";

    try {
        const response = await fetch(`${API_BASE}/api/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        
        if (response.ok) {
            localStorage.setItem('userId', result.user_id || data.user_id);
            localStorage.setItem('userName', result.name || data.name);
            localStorage.setItem('userRole', result.role || data.role);
            window.location.href = 'dashboard.html';
        } else {
            // Display backend error detail (e.g., "Duplicate entry")
            alert(result.detail || "Authentication Failure");
        }
    } catch (err) {
        alert("Node Communication Error: Ensure the FastAPI server is active.");
    }
});
