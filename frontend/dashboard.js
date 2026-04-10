/**
 * HealthBridge | Clinical Command Center Logic V9.1
 * Features: Role‑based default section, active link styling, no overview for patients.
 */

const API_BASE = window.location.origin;

document.addEventListener('DOMContentLoaded', () => {
    const userName = localStorage.getItem('userName') || 'Medical Professional';
    const userId = localStorage.getItem('userId');
    const userRole = localStorage.getItem('userRole') || 'patient';

    document.getElementById('displayUserName').innerText = userName;
    document.getElementById('displayUserId').innerText = `ID: ${userId}`;
    document.getElementById('roleBadge').innerText = userRole.toUpperCase();

    if (userRole === 'doctor') {
        document.getElementById('doctorLinks').classList.remove('hidden');
        document.getElementById('doctorStats').classList.remove('hidden');
        document.getElementById('doctorPatientList').classList.remove('hidden');
        document.getElementById('patientLinks').classList.add('hidden');
        loadDoctorOverview();
        // Default section for doctor: Overview
        showSection('overview');
    } else {
        document.getElementById('patientStats').classList.remove('hidden');
        // Default section for patient: AI Predictor
        showSection('predictor');
        loadPatientOverview();
        loadReports();
        loadPatientHealthTrend();
    }

    // AI Predictor form submission
    document.getElementById('predictForm').addEventListener('submit', handlePredict);
});

// --- SECTION TOGGLER (updates active link and title) ---
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(s => s.classList.add('hidden'));
    // Show selected section
    document.getElementById(sectionId).classList.remove('hidden');
    // Update title
    const titles = {
        'overview': 'Overview',
        'predictor': 'AI Predictor',
        'reports': 'My Reports',
        'manage-patients': 'Patient Checkup'
    };
    document.getElementById('sectionTitle').innerText = titles[sectionId] || sectionId;

    // Update active class on nav links
    document.querySelectorAll('.sidebar-nav .nav-link').forEach(link => {
        link.classList.remove('active');
    });
    // Find the link that calls this section (onclick attribute contains sectionId)
    document.querySelectorAll('.sidebar-nav .nav-link').forEach(link => {
        if (link.getAttribute('onclick') && link.getAttribute('onclick').includes(`'${sectionId}'`)) {
            link.classList.add('active');
        }
    });

    // Special handling for patient checkup to load patient lists
    if (sectionId === 'manage-patients') {
        loadAllPatients('A');
        loadAllPatients('B');
    }
}

// --- PATIENT OVERVIEW (with latest disease & risk) ---
async function loadPatientOverview() {
    const patientId = localStorage.getItem('userId');
    try {
        const response = await fetch(`${API_BASE}/api/reports/${patientId}`);
        const data = await response.json();
        
        document.getElementById('totalVisits').innerText = data.length;

        // Latest disease & risk
        if (data.length > 0) {
            const latest = data[0];
            document.getElementById('latestDisease').innerText = latest.result_status;
            document.getElementById('latestRisk').innerText = latest.analysis_score;
        } else {
            document.getElementById('latestDisease').innerText = '—';
            document.getElementById('latestRisk').innerText = '—';
        }

        // Disease frequency
        const diseaseCounts = {};
        data.forEach(report => {
            const disease = report.result_status;
            diseaseCounts[disease] = (diseaseCounts[disease] || 0) + 1;
        });

        const rankList = document.getElementById('diseaseRank');
        rankList.innerHTML = Object.entries(diseaseCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([name, count]) => `
                <div style="font-size: 0.85rem; padding: 5px 0; border-bottom: 1px solid #f1f5f9;">
                    <strong>${name}</strong>: ${count} times
                </div>
            `).join('');

    } catch (err) { console.error("Patient insights failed", err); }
}

// --- DOCTOR OVERVIEW (with patient list) ---
async function loadDoctorOverview() {
    try {
        const response = await fetch('${API_BASE}/api/doctor-stats');
        const stats = await response.json();

        document.getElementById('totalPatients').innerText = stats.total_patients;
        document.getElementById('totalChecks').innerText = stats.total_checks;
        document.getElementById('respondedCount').innerText = stats.responded;
        document.getElementById('pendingCount').innerText = stats.pending;

        // Load patient list with latest risk
        const patientRes = await fetch('${API_BASE}/api/doctor-patient-list');
        const patients = await patientRes.json();
        const container = document.getElementById('patientListContainer');
        container.innerHTML = patients.map(p => `
            <div class="patient-list-item">
                <div>
                    <strong>${p.name}</strong> (${p.id_str})<br>
                    <small>Latest: ${p.latest_disease} ${p.latest_risk ? '('+p.latest_risk+'%)' : ''}</small>
                </div>
                ${p.latest_risk ? `<span class="risk-badge">Risk ${p.latest_risk}%</span>` : '<span class="no-data">No data</span>'}
            </div>
        `).join('');
    } catch (err) { console.error("Doctor insights failed", err); }
}

// --- MY REPORTS ---
async function loadReports() {
    const patientId = localStorage.getItem('userId');
    const reportList = document.getElementById('patientReportList');
    
    try {
        const feedbackRes = await fetch(`${API_BASE}/api/get-feedback/${patientId}`);
        const feedback = await feedbackRes.json();
        if (feedback && feedback.message) {
            document.getElementById('doctorMessageArea').classList.remove('hidden');
            document.getElementById('feedbackText').innerText = `"${feedback.message}"`;
        }

        const response = await fetch(`${API_BASE}/api/reports/${patientId}`);
        const reports = await response.json();
        
        reportList.innerHTML = reports.map(r => `
            <div class="summary-card" style="margin-bottom: 1rem; border-left: 5px solid #8686AC;">
                <h4 style="color: #0A0A23;">${r.result_status} (Risk: ${r.analysis_score}%)</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                    <div style="background: #F1F5F9; padding: 10px; border-radius: 8px;">
                        <strong>Drugs:</strong><br><small>${r.suggested_drugs}</small>
                    </div>
                    <div style="background: #F1F5F9; padding: 10px; border-radius: 8px;">
                        <strong>Diet:</strong><br><small>${r.suggested_foods}</small>
                    </div>
                </div>
                <p style="font-size: 0.7rem; margin-top: 8px; color: #64748B;">Date: ${new Date(r.created_at).toLocaleString()}</p>
            </div>
        `).join('');
    } catch (err) { console.error("Reports loading failed", err); }
}

// --- AI PREDICTOR HANDLER ---
async function handlePredict(e) {
    e.preventDefault();
    const userId = localStorage.getItem('userId');
    if (!userId) return alert("Please log in first.");

    const inputData = {
        user_id: userId,
        temperature: parseFloat(document.getElementById('temperature').value),
        heart_rate: parseFloat(document.getElementById('heart_rate').value),
        bp_sys: parseFloat(document.getElementById('bp_sys').value),
        bp_dia: parseFloat(document.getElementById('bp_dia').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        fever: parseInt(document.getElementById('fever').value),
        cough: parseInt(document.getElementById('cough').value),
        chest_pain: parseInt(document.getElementById('chest_pain').value),
        shortness_breath: parseInt(document.getElementById('shortness_breath').value),
        fatigue: parseInt(document.getElementById('fatigue').value),
        headache: parseInt(document.getElementById('headache').value)
    };

    try {
        const res = await fetch('${API_BASE}/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });
        const result = await res.json();
        if (res.ok) {
            document.getElementById('predictionResult').classList.remove('hidden');
            document.getElementById('predDisease').innerText = result.prediction;
            document.getElementById('predScore').innerText = result.score;
            document.getElementById('predDrugs').innerText = result.drugs;
            document.getElementById('predFoods').innerText = result.foods;
            document.getElementById('predRoutine').innerText = result.routine;
            // Refresh reports and overview after new prediction
            loadReports();
            loadPatientOverview();
            loadPatientHealthTrend();
        } else {
            alert("Prediction failed: " + result.detail);
        }
    } catch (err) {
        console.error(err);
        alert("Network error.");
    }
}

// --- DOCTOR: LOAD ALL PATIENTS INTO LIST ---
async function loadAllPatients(side) {
    try {
        const res = await fetch('${API_BASE}/api/search-patient?q='); // empty query returns all
        const list = await res.json();
        document.getElementById(`list${side}`).innerHTML = list.map(p => `
            <div class="registry-item" onclick="selectPatient('${side}', '${p.id_str}', '${p.name}')">
                <strong>${p.name}</strong><br><small>${p.id_str}</small>
            </div>
        `).join('');
    } catch (err) { console.error(err); }
}

// --- DOCTOR: PATIENT SELECTION & GRAPHS ---
async function selectPatient(side, id_str, name) {
    document.getElementById(`data${side}`).classList.remove('hidden');
    document.getElementById(`name${side}`).innerText = `${name} // ID: ${id_str}`;
    document.getElementById(`list${side}`).innerHTML = ""; 

    try {
        const response = await fetch(`${API_BASE}/api/reports/${id_str}`);
        const data = await response.json();
        if (data.length > 0) {
            renderPatientCharts(side, data);
            
            document.getElementById(`history${side}`).innerHTML = data.map(r => `
                <div style="font-size: 0.75rem; border-bottom: 1px solid #EEE; padding: 5px 0;">
                    <strong>${r.result_status}</strong> (${new Date(r.created_at).toLocaleDateString()})<br>
                    <small>Meds: ${r.suggested_drugs}</small><br>
                    <small>Food: ${r.suggested_foods}</small>
                </div>
            `).join('');
        }
    } catch (err) { console.error("Vital loading failed", err); }
}

function renderPatientCharts(side, data) {
    // Destroy existing charts if any (simplified: assume new Chart each time)
    const labels = data.slice(0, 5).reverse().map((_, i) => `T-${i}`);

    // Temperature
    new Chart(document.getElementById(`chart${side}_temp`), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{ label: 'Temp °C', data: data.slice(0, 5).reverse().map(r => r.temperature), borderColor: '#FF6B6B' }]
        }
    });

    // BP
    new Chart(document.getElementById(`chart${side}_bp`), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'SYS', data: data.slice(0, 5).reverse().map(r => r.bp_sys), borderColor: '#4ECDC4' },
                { label: 'DIA', data: data.slice(0, 5).reverse().map(r => r.bp_dia), borderColor: '#45B7D1' }
            ]
        }
    });

    // Heart Rate
    new Chart(document.getElementById(`chart${side}_hr`), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{ label: 'BPM', data: data.slice(0, 5).reverse().map(r => r.heart_rate), borderColor: '#8686AC' }]
        }
    });

    // Symptom Frequency (count of occurrences where value > 0)
    const symptomNames = ['Fever', 'Cough', 'Chest Pain', 'Shortness Breath', 'Fatigue', 'Headache'];
    const counts = [0,0,0,0,0,0];
    data.forEach(r => {
        if (r.fever > 0) counts[0]++;
        if (r.cough > 0) counts[1]++;
        if (r.chest_pain > 0) counts[2]++;
        if (r.shortness_breath > 0) counts[3]++;
        if (r.fatigue > 0) counts[4]++;
        if (r.headache > 0) counts[5]++;
    });

    new Chart(document.getElementById(`chart${side}_symptoms`), {
        type: 'bar',
        data: {
            labels: symptomNames,
            datasets: [{
                label: 'Number of visits with symptom',
                data: counts,
                backgroundColor: '#8686AC'
            }]
        }
    });
}

// --- DOCTOR: LIVE SEARCH (loads all if query too short) ---
async function liveSearch(side) {
    const q = document.getElementById(`input${side}`).value;
    if (q.length < 2) {
        loadAllPatients(side);
        return;
    }
    try {
        const res = await fetch(`${API_BASE}/api/search-patient?q=${q}`);
        const list = await res.json();
        document.getElementById(`list${side}`).innerHTML = list.map(p => `
            <div class="registry-item" onclick="selectPatient('${side}', '${p.id_str}', '${p.name}')">
                <strong>${p.name}</strong><br><small>${p.id_str}</small>
            </div>
        `).join('');
    } catch (err) { console.error(err); }
}

// --- DOCTOR: SEND MESSAGE ---
async function sendAdvice(side) {
    const pId = document.getElementById(`name${side}`).innerText.split('ID: ')[1];
    const msg = document.getElementById(`msg${side}`).value;
    if (!msg) return alert("Please type a message.");

    try {
        const response = await fetch('${API_BASE}/api/send-feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ doctor_id: localStorage.getItem('userId'), patient_id: pId, message: msg })
        });
        if (response.ok) {
            alert("Message sent to patient.");
            document.getElementById(`msg${side}`).value = "";
            loadDoctorOverview();
        }
    } catch (err) { alert("Sending failed."); }
}

// --- PATIENT HEALTH TREND CHART ---
async function loadPatientHealthTrend() {
    const patientId = localStorage.getItem('userId');
    try {
        const res = await fetch(`${API_BASE}/api/reports/${patientId}`);
        const reports = await res.json();
        const recent = reports.slice(0, 7).reverse(); // last 7 in chronological order
        new Chart(document.getElementById('healthChart'), {
            type: 'line',
            data: {
                labels: recent.map((_, i) => `Visit ${i+1}`),
                datasets: [{
                    label: 'Risk %',
                    data: recent.map(r => r.analysis_score),
                    borderColor: '#8686AC',
                    backgroundColor: 'rgba(134,134,172,0.1)'
                }]
            }
        });
    } catch (err) { console.error(err); }
}
