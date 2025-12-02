# 📊 ProfitLens — Business Data Analyzer

A powerful all-in-one **data cleaning, analysis, visualization, and reporting** tool built with **Streamlit**, integrating:

* Automated column inference (LLM-powered)
* Advanced data cleaning & validation pipeline
* 30+ business analytics functions (sales, customers, trends)
* 25+ visualizations (matplotlib, seaborn)
* Optional login system with MySQL backend
* Complete Streamlit UI with multi-tab dashboards

---

# 📁 Project Structure

```
📦 ProfitLens
│
├── analysis.py               # Business analytics functions
├── visualization.py          # All Matplotlib/Seaborn visualizations
├── cleaner.py                # LLM + Rule-based data cleaning
├── llmutil.py                # Gemini integration for LLM tasks
├── login_system.py           # MySQL-based authentication
├── app_updated.py / app2.py  # Main Streamlit app
├── .env                      # Environment variables
└── README.md                 # Documentation
```

---

# ⚙️ Setup Instructions

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ProfitLens.git
cd ProfitLens
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4️⃣ Create and Configure `.env` File

Create a `.env` file in the project root:

```
# -------- GEMINI API KEY --------
API_KEY=your_gemini_key_here
GEMINI_API_KEY=your_gemini_key_here

# -------- DATABASE CONFIG --------
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=yourpassword
DB_NAME=profitlens
```

### 🔹 Used For

* `API_KEY` — Column inference, summary generation, insights
* `DB_*` — MySQL login system

If no DB details are provided → Login system automatically switches to **Guest Mode**.

---

# 5️⃣ Initialize the Database (Optional)

If using login authentication:

```sql
CREATE DATABASE profitlens;
```

Or auto-create tables by running:

```python
from login_system import init_database
init_database()
```

---

# 6️⃣ Run the App

```bash
streamlit run app.py
```



---

# 🚀 How to Use ProfitLens

## 🔧 Step 1: Upload Data

Upload your CSV/XLSX via the sidebar.
A preview of first 5 rows will appear.

## 🔧 Step 2: Column Mapping

* Auto-detects common column names
* You can manually map incorrect columns
* LLM-based inference when enabled

Click **Apply Mapping & Clean Data**.

## 🔧 Step 3: Automated Cleaning

Cleaner performs:

* Column renaming
* Data type correction
* Null handling
* Age/Date/Price/Quantity validation
* Removes duplicate transaction IDs
* Auto-calculates Total Amount

Cleaned dataset loads into analysis.

---

# 📊 Dashboard Regions

## 🛍 Product Analysis

* Top products by revenue/quantity
* Monthly performance
* Price elasticity

## 👥 Customer Analysis

* Top customers
* New vs returning
* RFM segmentation
* CLV

## 📅 Trends

* Sales trend
* Growth rate
* Moving average
* Forecasting
* Anomaly detection

## 📂 Category Breakdown

* Category revenue share

## 💰 Profit & Inventory

* Total profit & margin
* Stock-out risk
* Reorder point suggestions

## 📤 Export

* Download cleaned dataset
* Generate HTML report

---

# 🙌 Contributing

PRs are welcome.

---


MIT License.
