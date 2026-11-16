# data/

The **UCI Online Shoppers Purchasing Intention Dataset (ID 468)** lives here.

Download it (from the `backend` folder):

```powershell
python -I ml/download_data.py
```

| File | Purpose | In git? |
|---|---|---|
| `online_shoppers_intention.csv` | Original dataset. **Never edited**; code only reads it | No |
| `online_shoppers_intention.sha256` | Checksum recorded at download. `load_raw()` verifies it | Yes |
| `raw/online_shoppers_purchasing_intention_dataset.zip` | Original archive from UCI | No |

Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
(License: CC BY 4.0. Sakar, C. & Kastro, Y., 2018.)
