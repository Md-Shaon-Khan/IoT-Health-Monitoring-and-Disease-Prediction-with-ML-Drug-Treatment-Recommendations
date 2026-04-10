/**
 * AI HEALTH PREDICTOR LOGIC
 * Features: 11 (Temperature, Heart Rate, Diastolic BP, Systolic BP, Humidity, + 6 Symptoms)
 * Model Shape: Expected 11 inputs
 */

const API_BASE = window.location.origin;

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // ১. ফর্ম থেকে ১১টি ফিচারের ডাটা সংগ্রহ করা
    const formData = {
    user_id: localStorage.getItem('userId'),
    temperature: parseFloat(document.getElementById('temp').value), // 'temp' এর বদলে 'temperature'
    heart_rate: parseFloat(document.getElementById('heart_rate').value),
    bp_dia: parseFloat(document.getElementById('bp_dia').value),
    bp_sys: parseFloat(document.getElementById('bp_sys').value),
    humidity: parseFloat(document.getElementById('humidity').value),
    fever: document.getElementById('fever').checked ? 1 : 0,
    cough: document.getElementById('cough').checked ? 1 : 0,
    chest_pain: document.getElementById('chest_pain').checked ? 1 : 0,
    shortness_breath: document.getElementById('shortness_breath').checked ? 1 : 0,
    fatigue: document.getElementById('fatigue').checked ? 1 : 0,
    headache: document.getElementById('headache').checked ? 1 : 0
};

    // ২. ডিবাগিং: কনসোলে চেক করা ডাটা ঠিক আছে কি না
    console.log("Submitting 11 Features for Prediction:", formData);

    try {
        // ৩. ব্যাকএন্ড এপিআই (FastAPI) কল করা
        const response = await fetch('${API_BASE}/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        // ৪. সফলভাবে প্রেডিকশন রেজাল্ট পাওয়া গেলে UI আপডেট করা
        if (response.ok) {
            // রেজাল্ট বক্সের টেক্সট পরিবর্তন
            document.getElementById('predictionResult').innerText = "Predicted Condition: " + result.prediction;
            document.getElementById('predictionScore').innerText = "Health Score: " + result.score + "%";
            
            // রেজাল্ট বক্সটি দৃশ্যমান করা
            const resultBox = document.getElementById('resultBox');
            resultBox.style.display = 'block';
            
            // স্মুথলি রেজাল্ট সেকশনে স্ক্রল করা
            window.scrollTo({
                top: resultBox.offsetTop - 20,
                behavior: 'smooth'
            });
        } else {
            // ব্যাকএন্ড থেকে আসা এরর মেসেজ দেখানো
            alert("Error from Server: " + (result.detail || "Prediction failed"));
        }

    } catch (err) {
        // সার্ভার বন্ধ থাকলে বা কানেকশন না পেলে এরর
        console.error("Fetch Error:", err);
        alert("Server not responding. Please make sure app.py is running via uvicorn.");
    }
});
