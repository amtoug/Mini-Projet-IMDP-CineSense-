const API_URL = "http://127.0.0.1:8000";

function showResult(elementId, sentimentType, sentimentValue, scoreInfo) {
    const resultEl = document.getElementById(elementId);
    resultEl.className = "result";

    let emoji = "";
    let colorClass = "";

    // Gestion des sentiments
    if (sentimentType === "vader") {
        if (sentimentValue === 1) {
            emoji = "😊";
            colorClass = "positive";            
            value = "positive";
        } else if (sentimentValue === -1) {
            emoji = "😠";
            colorClass = "negative";            
            value = "negative";
        } else {
            emoji = "😐";
            colorClass = "neutral";            
            value = "neutral";
        }
        resultEl.innerHTML = `
            
            <strong>VADER Sentiment:</strong><br> ${value}<span>${emoji}</span><br>
            <pre>${JSON.stringify(scoreInfo, null, 2)}</pre>
        `;
    } else if (sentimentType === "bert") {
        emoji = sentimentValue === "POSITIVE" ? "😍" : "😡";
        colorClass = sentimentValue === "POSITIVE" ? "positive" : "negative";
        resultEl.innerHTML = `
            
            <strong>DistilBERT:</strong> <br>${sentimentValue}<span>${emoji}</span><br>
            Confidence: ${(scoreInfo * 100).toFixed(2)}%
        `;
    }

    resultEl.classList.add(colorClass);
}

// Analyse VADER
async function analyzeVader() {
    const text = document.getElementById("textInput").value.trim();
    if (!text) return alert("Veuillez entrer un texte.");

    const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text})
    });

    const data = await response.json();
    showResult("result", "vader", data.sentiment, data.scores);
}

// Analyse BERT
async function analyzeBert() {
    const text = document.getElementById("textInput").value.trim();
    if (!text) return alert("Veuillez entrer un texte.");

    const response = await fetch(`${API_URL}/predict_bert`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text})
    });

    const data = await response.json();
    showResult("result", "bert", data.label, data.score);
}
