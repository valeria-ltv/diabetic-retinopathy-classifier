# Progress Log

## 15-09-2026 — Step 1: Environment setup
- Created GitHub repo, folder structure
- Downloaded APTOS-2019 via Kaggle API
- Verified GPU access, sanity-checked data loading

## 16-09-2026 — Step 2: EDA
- Checked data integrity (no duplicates, no missing values)
- Class distribution: severe imbalance, classes 0/2 dominant
- Image sizes vary → need padding + resize
- Two visually distinct image types (color balance) → possible camera/source difference
- ~~TODO (next step): verify whether image type correlates with diagnosis class before preprocessing~~ → done below (2026-09-21)

## 21-09-2026 — Color bias check (closed TODO)
- Verified "cool" vs "warm" image correlation with diagnosis using HSV saturation
- Confirmed: class 0 has the highest proportion of cool images — risk of shortcut learning
- Requirement for preprocessing: aggressive color normalization

## Next: Step 3 — Preprocessing & augmentation pipeline