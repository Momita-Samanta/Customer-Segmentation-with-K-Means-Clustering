# 🧠 Customer Segmentation with K-Means Clustering

> Group customers by behaviour. Find patterns. Drive action.  
> **30 minutes. Real data. Real insight.**

---

## 📌 Overview

This project segments customers using **RFM analysis** combined with **K-Means clustering** to identify behavioural patterns and unlock targeted marketing strategies.

Each customer is represented by three numbers:

| Feature | Description |
|---|---|
| **R**ecency | Days since last purchase |
| **F**requency | Total orders placed |
| **M**onetary | Total spend |

---

## 🗂️ Project Structure

```
customer-segmentation/
├── data/
│   └── online_retail.csv        # Raw transactional data
├── notebooks/
│   └── segmentation.ipynb       # Main analysis notebook
├── src/
│   ├── preprocess.py            # Data cleaning & RFM feature engineering
│   ├── cluster.py               # K-Means model & elbow curve
│   └── visualise.py             # Segment visualisation
├── outputs/
│   └── customer_segments.csv    # Final labelled output
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/Momita-Samanta/customer-segmentation-with-K-Means-Clustering.git
cd customer-segmentation

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 🚀 Pipeline

### Step 1 — Clean the Data

Remove noise so your clusters are based on real behaviour.

```python
df = pd.read_csv("data/online_retail.csv", encoding="ISO-8859-1")

# Drop rows with missing CustomerID
df.dropna(subset=["CustomerID"], inplace=True)

# Remove cancellations (invoices starting with 'C')
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Create Revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
```

---

### Step 2 — Build RFM Features

One row per customer. Three columns. That's your model input.

```python
import datetime

snapshot_date = df["InvoiceDate"].max() + datetime.timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency   = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency = ("InvoiceNo",   "nunique"),
    Monetary  = ("Revenue",     "sum")
).reset_index()
```

---

### Step 3 — Scale the Data

Log-transform to reduce skew, then standardise. Skip this and your clusters will be meaningless.

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
scaler  = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
```

---

### Step 4 — Find k (Elbow Method)

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)

plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Curve")
plt.show()
```

> 🔍 Look for the "elbow" — the point where inertia stops dropping sharply. Typically **k = 3 or 4** works well for RFM data.

---

### Step 5 — Run the Model & Label Segments

```python
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm["Cluster"] = km.fit_predict(rfm_scaled)

# Inspect cluster centres to label them
print(rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean())
```

Map clusters to human-readable segment names based on the centre values:

| Segment | Recency | Spend | Profile |
|---|---|---|---|
| 🏆 **Champions** | Low | High | Recent, frequent, big spenders |
| 🌱 **New Customers** | Low | Low | Recent but haven't spent much yet |
| ⚠️ **At Risk** | High | High | Used to spend big — going quiet |
| 💤 **Lost** | High | Low | Haven't bought in a while, low value |

```python
segment_map = {
    0: "Champions",
    1: "New Customers",
    2: "At Risk",
    3: "Lost"
}
# Adjust mapping after inspecting your cluster centres
rfm["Segment"] = rfm["Cluster"].map(segment_map)
```

---

### Step 6 — Write Your Insight

```python
segment_counts = rfm["Segment"].value_counts(normalize=True) * 100
print(segment_counts.round(1))
```

**Example output:**
```
At Risk          23.1%
Champions        31.4%
New Customers    28.7%
Lost             16.8%
```

> 💡 *"23% of customers are At Risk — they've spent big before but are going quiet. Time for a win-back campaign."*

---

## 📊 Sample Output

```
CustomerID  Recency  Frequency  Monetary   Segment
12346       326      1          77183.60   Lost
12347       2        7          4310.00    Champions
12348       75       4          1797.24    At Risk
12350       310      1          334.40     Lost
12352       36       8          2506.04    Champions
```

---

## 📈 Visualisation

```python
import seaborn as sns

sns.scatterplot(
    data=rfm,
    x="Recency",
    y="Monetary",
    hue="Segment",
    palette="Set2",
    alpha=0.7
)
plt.title("Customer Segments: Recency vs Monetary Value")
plt.show()
```

---

## 🔁 Extending This Project

- **Add more features** — average order value, product category preferences
- **Try different k values** — use silhouette score alongside the elbow method
- **Automate segment labelling** — use cluster centile ranks instead of manual mapping
- **Schedule it** — re-run monthly to track segment migration over time
- **Connect to CRM** — push segment labels back into your marketing platform

---

## 📄 License

MIT — use it, fork it, ship it.

---

## 🙌 Acknowledgements

Dataset: [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)  
Technique: RFM Analysis + K-Means Clustering
