from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.schema import ChurnResponse, CustomerFeatures

PREPROCESSOR_PATH = Path("artifacts/data_transformation/preprocessor.joblib")
MODEL_PATH = Path("artifacts/model_trainer/model.joblib")

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predicts whether a customer will churn, using the trained model.",
    version="1.0.0",
)

# load artifacts once at startup (the SAME preprocessor used in training, so
# inference applies identical encoding/scaling)
if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
    raise RuntimeError(
        "Model artifacts not found. Run `python main.py` first to generate "
        f"'{PREPROCESSOR_PATH}' and '{MODEL_PATH}'."
    )
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Customer Churn Risk</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet" />
<style>
  :root{
    --bg:#EAEEF7; --bg2:#F7F9FE; --surface:#FFFFFF;
    --ink:#151B2B; --muted:#5B6580; --line:#E3E8F2;
    --brand:#5B54E6; --brand-press:#4842C9;
    --green:#159A48; --amber:#DE8400; --red:#DA2A2A;
    --display:'Space Grotesk',system-ui,-apple-system,sans-serif;
    --body:'Plus Jakarta Sans',system-ui,-apple-system,sans-serif;
  }
  *{box-sizing:border-box}
  body{
    margin:0; min-height:100vh; font-family:var(--body); color:var(--ink);
    background:radial-gradient(1200px 600px at 50% -10%, var(--bg2), var(--bg));
    display:grid; place-items:start center; padding:32px 18px 64px;
  }
  .card{
    width:100%; max-width:780px; background:var(--surface);
    border:1px solid var(--line); border-radius:22px; overflow:hidden;
    box-shadow:0 24px 60px -28px rgba(21,27,43,.30), 0 2px 6px rgba(21,27,43,.04);
  }
  .topbar{height:5px; background:linear-gradient(90deg,var(--green),var(--amber),var(--red));}
  .head{padding:28px 30px 8px;}
  .eyebrow{font-family:var(--display); font-size:12px; font-weight:600; letter-spacing:.14em;
    text-transform:uppercase; color:var(--brand);}
  .head h1{font-family:var(--display); font-weight:700; font-size:30px; line-height:1.1;
    margin:8px 0 6px; letter-spacing:-.01em;}
  .head p{margin:0; color:var(--muted); font-size:15px; max-width:54ch;}
  form{padding:14px 30px 4px;}
  .section{padding:18px 0; border-top:1px solid var(--line);}
  .section:first-of-type{border-top:none;}
  .seclabel{font-family:var(--display); font-size:12px; font-weight:600; letter-spacing:.1em;
    text-transform:uppercase; color:var(--muted); margin:0 0 14px;}
  .grid{display:grid; grid-template-columns:repeat(2,1fr); gap:14px 18px;}
  .field{display:flex; flex-direction:column; min-width:0;}
  .field label{font-size:13px; font-weight:600; margin-bottom:6px; color:var(--ink);}
  select,input{
    font-family:var(--body); font-size:14px; color:var(--ink); background:#FBFCFE;
    border:1px solid var(--line); border-radius:11px; padding:11px 12px; width:100%;
    appearance:none; -webkit-appearance:none; transition:border-color .15s, box-shadow .15s;
  }
  select{
    padding-right:36px; cursor:pointer;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%235B6580' stroke-width='1.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    background-repeat:no-repeat; background-position:right 13px center;
  }
  select:focus,input:focus{outline:none; border-color:var(--brand);
    box-shadow:0 0 0 3px rgba(91,84,230,.18);}
  .actions{padding:8px 0 26px;}
  .cta{
    width:100%; border:none; cursor:pointer; color:#fff; background:var(--brand);
    font-family:var(--display); font-weight:600; font-size:15px; letter-spacing:.01em;
    padding:15px 18px; border-radius:13px; transition:background .15s, transform .05s;
  }
  .cta:hover{background:var(--brand-press);}
  .cta:active{transform:translateY(1px);}
  .cta:disabled{opacity:.6; cursor:default;}
  .result{display:none; padding:6px 30px 34px;}
  .result.show{display:block; animation:rise .45s ease both;}
  @keyframes rise{from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;}}
  .gaugewrap{display:flex; flex-direction:column; align-items:center; text-align:center;}
  svg.gauge{width:300px; max-width:100%; height:auto; display:block;}
  .gnum{font-family:var(--display); font-weight:700; font-size:46px; line-height:1; margin-top:2px;}
  .verdict{font-family:var(--display); font-weight:600; font-size:18px; margin-top:6px;}
  .vsub{color:var(--muted); font-size:14.5px; margin-top:8px; max-width:46ch;}
  .endlabels{display:flex; justify-content:space-between; width:300px; max-width:100%;
    margin-top:-6px; color:var(--muted); font-size:12px; font-weight:600;
    font-family:var(--display); letter-spacing:.04em;}
  .error{display:none; margin:0 30px 30px; padding:14px 16px; border-radius:12px;
    background:#FDEDED; border:1px solid #F6CFCF; color:#A81E1E; font-size:14px;}
  .error.show{display:block;}
  .foot{padding:0 30px 26px; color:var(--muted); font-size:12.5px;}
  .foot a{color:var(--brand); text-decoration:none;}
  .foot a:hover{text-decoration:underline;}
  @media (max-width:560px){
    .grid{grid-template-columns:1fr;}
    .head h1{font-size:25px;}
    .head,.head{padding-left:22px; padding-right:22px;}
    form,.result,.foot,.error{padding-left:22px; padding-right:22px;}
    .error{margin-left:22px; margin-right:22px;}
  }
  @media (prefers-reduced-motion:reduce){
    .result.show{animation:none;}
    *{transition:none !important;}
  }
</style>
</head>
<body>
  <main class="card">
    <div class="topbar"></div>
    <div class="head">
      <div class="eyebrow">Retention tool</div>
      <h1>Customer churn risk</h1>
      <p>Enter a customer's plan details and see how likely they are to leave. The needle sweeps from loyal to at&nbsp;risk.</p>
    </div>

    <form id="form">
      <div class="section">
        <p class="seclabel">Customer</p>
        <div class="grid">
          <div class="field">
            <label for="gender">Gender</label>
            <select id="gender" name="gender">
              <option value="Female" selected>Female</option>
              <option value="Male">Male</option>
            </select>
          </div>
          <div class="field">
            <label for="SeniorCitizen">Senior citizen</label>
            <select id="SeniorCitizen" name="SeniorCitizen">
              <option value="0" selected>No</option>
              <option value="1">Yes</option>
            </select>
          </div>
          <div class="field">
            <label for="Partner">Has a partner</label>
            <select id="Partner" name="Partner">
              <option value="Yes" selected>Yes</option>
              <option value="No">No</option>
            </select>
          </div>
          <div class="field">
            <label for="Dependents">Has dependents</label>
            <select id="Dependents" name="Dependents">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
            </select>
          </div>
        </div>
      </div>

      <div class="section">
        <p class="seclabel">Account</p>
        <div class="grid">
          <div class="field">
            <label for="tenure">Months as customer</label>
            <input id="tenure" name="tenure" type="number" min="0" max="100" step="1" value="5" />
          </div>
          <div class="field">
            <label for="Contract">Contract type</label>
            <select id="Contract" name="Contract">
              <option value="Month-to-month" selected>Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>
          </div>
          <div class="field">
            <label for="PaperlessBilling">Paperless billing</label>
            <select id="PaperlessBilling" name="PaperlessBilling">
              <option value="Yes" selected>Yes</option>
              <option value="No">No</option>
            </select>
          </div>
          <div class="field">
            <label for="PaymentMethod">Payment method</label>
            <select id="PaymentMethod" name="PaymentMethod">
              <option value="Electronic check" selected>Electronic check</option>
              <option value="Mailed check">Mailed check</option>
              <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
              <option value="Credit card (automatic)">Credit card (automatic)</option>
            </select>
          </div>
        </div>
      </div>

      <div class="section">
        <p class="seclabel">Services</p>
        <div class="grid">
          <div class="field">
            <label for="PhoneService">Phone service</label>
            <select id="PhoneService" name="PhoneService">
              <option value="Yes" selected>Yes</option>
              <option value="No">No</option>
            </select>
          </div>
          <div class="field">
            <label for="MultipleLines">Multiple lines</label>
            <select id="MultipleLines" name="MultipleLines">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No phone service">No phone service</option>
            </select>
          </div>
          <div class="field">
            <label for="InternetService">Internet service</label>
            <select id="InternetService" name="InternetService">
              <option value="Fiber optic" selected>Fiber optic</option>
              <option value="DSL">DSL</option>
              <option value="No">No internet</option>
            </select>
          </div>
          <div class="field">
            <label for="OnlineSecurity">Online security</label>
            <select id="OnlineSecurity" name="OnlineSecurity">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div class="field">
            <label for="OnlineBackup">Online backup</label>
            <select id="OnlineBackup" name="OnlineBackup">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div class="field">
            <label for="DeviceProtection">Device protection</label>
            <select id="DeviceProtection" name="DeviceProtection">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div class="field">
            <label for="TechSupport">Tech support</label>
            <select id="TechSupport" name="TechSupport">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div class="field">
            <label for="StreamingTV">Streaming TV</label>
            <select id="StreamingTV" name="StreamingTV">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div class="field">
            <label for="StreamingMovies">Streaming movies</label>
            <select id="StreamingMovies" name="StreamingMovies">
              <option value="No" selected>No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
        </div>
      </div>

      <div class="section">
        <p class="seclabel">Charges</p>
        <div class="grid">
          <div class="field">
            <label for="MonthlyCharges">Monthly charges ($)</label>
            <input id="MonthlyCharges" name="MonthlyCharges" type="number" min="0" step="0.05" value="85.70" />
          </div>
          <div class="field">
            <label for="TotalCharges">Total charges ($)</label>
            <input id="TotalCharges" name="TotalCharges" type="number" min="0" step="0.05" value="428.50" />
          </div>
        </div>
      </div>

      <div class="actions">
        <button class="cta" id="submit" type="submit">Check churn risk</button>
      </div>
    </form>

    <div class="error" id="error"></div>

    <div class="result" id="result">
      <div class="gaugewrap">
        <svg class="gauge" viewBox="0 0 300 176" role="img" aria-label="Churn risk gauge">
          <defs>
            <linearGradient id="riskgrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#159A48" />
              <stop offset="50%" stop-color="#DE8400" />
              <stop offset="100%" stop-color="#DA2A2A" />
            </linearGradient>
          </defs>
          <path d="M40 152 A110 110 0 0 1 260 152" fill="none" stroke="#EAEEF7" stroke-width="22" stroke-linecap="round" />
          <path d="M40 152 A110 110 0 0 1 260 152" fill="none" stroke="url(#riskgrad)" stroke-width="14" stroke-linecap="round" />
          <g id="needle" transform="rotate(-90 150 152)">
            <polygon points="145,152 150,58 155,152" fill="#151B2B" />
          </g>
          <circle cx="150" cy="152" r="9" fill="#151B2B" />
          <circle cx="150" cy="152" r="3.5" fill="#fff" />
        </svg>
        <div class="endlabels"><span>Loyal</span><span>At risk</span></div>
        <div class="gnum" id="gnum">—</div>
        <div class="verdict" id="verdict"></div>
        <div class="vsub" id="vsub"></div>
      </div>
    </div>

    <div class="foot">
      Predictions come from a logistic-regression model (ROC-AUC ≈ 0.84) trained on the IBM Telco dataset.
      <a href="https://github.com/akhilsai007/Telco-Churn-mlops" target="_blank" rel="noopener">View the project on GitHub</a>.
    </div>
  </main>

<script>
  const form = document.getElementById("form");
  const btn = document.getElementById("submit");
  const result = document.getElementById("result");
  const errBox = document.getElementById("error");
  const gnum = document.getElementById("gnum");
  const verdict = document.getElementById("verdict");
  const vsub = document.getElementById("vsub");
  const needle = document.getElementById("needle");

  const NUM_INT = ["tenure", "SeniorCitizen"];
  const NUM_FLOAT = ["MonthlyCharges", "TotalCharges"];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function buildPayload(){
    const data = {};
    new FormData(form).forEach((value, key) => {
      if (NUM_INT.includes(key)) data[key] = parseInt(value, 10);
      else if (NUM_FLOAT.includes(key)) data[key] = parseFloat(value);
      else data[key] = value;
    });
    return data;
  }

  function band(p){
    if (p < 0.34) return {c:"var(--green)", v:"Low risk",
      s:"This customer looks likely to stay."};
    if (p < 0.66) return {c:"var(--amber)", v:"Medium risk",
      s:"This customer could go either way \u2014 worth keeping an eye on."};
    return {c:"var(--red)", v:"High risk",
      s:"This customer is likely to leave \u2014 a strong candidate for a retention offer."};
  }

  function showResult(prob){
    const b = band(prob);
    gnum.style.color = b.c;
    verdict.style.color = b.c;
    verdict.textContent = b.v;
    vsub.textContent = b.s;
    result.classList.add("show");

    const targetDeg = 180 * prob - 90;
    if (reduceMotion){
      needle.setAttribute("transform", "rotate(" + targetDeg + " 150 152)");
      gnum.textContent = Math.round(prob * 100) + "%";
      return;
    }
    const startDeg = -90, dur = 900, t0 = performance.now();
    function frame(now){
      let k = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      const deg = startDeg + (targetDeg - startDeg) * e;
      needle.setAttribute("transform", "rotate(" + deg + " 150 152)");
      gnum.textContent = Math.round((0 + prob * e) * 100) + "%";
      if (k < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    errBox.classList.remove("show");
    btn.disabled = true;
    btn.textContent = "Checking\u2026";
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(buildPayload())
      });
      if (!res.ok){
        let detail = "The model rejected the input (status " + res.status + ").";
        try { const j = await res.json(); if (j && j.detail) detail = "Check your entries and try again."; } catch(e){}
        throw new Error(detail);
      }
      const data = await res.json();
      showResult(data.churn_probability);
      result.scrollIntoView({behavior: reduceMotion ? "auto" : "smooth", block: "nearest"});
    } catch (err) {
      result.classList.remove("show");
      errBox.textContent = (err && err.message)
        ? err.message
        : "Couldn't reach the model. Check your connection and try again.";
      errBox.classList.add("show");
    } finally {
      btn.disabled = false;
      btn.textContent = "Check churn risk";
    }
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=ChurnResponse)
def predict(features: CustomerFeatures):
    # one-row DataFrame with the same column names the preprocessor was fit on
    df = pd.DataFrame([features.model_dump()])
    X = preprocessor.transform(df)
    # restore feature names (the model was trained on named columns)
    X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())
    probability = float(model.predict_proba(X)[0, 1])
    label = "Yes" if probability >= 0.5 else "No"
    return ChurnResponse(churn=label, churn_probability=round(probability, 4))