# Progress Log

## 15-09-2026 — Step 1: Environment setup
- Created GitHub repo, folder structure
- Downloaded APTOS-2019 via Kaggle API
- Verified GPU access, sanity-checked data loading

## 15-09-2026 — Step 2: EDA
- Checked data integrity (no duplicates, no missing values)
- Class distribution: severe imbalance, classes 0/2 dominant
- Image sizes vary → need padding + resize
- Two visually distinct image types (color balance) → possible camera/source difference
- **TODO (next step):** verify whether image type correlates with diagnosis class before preprocessing — risk of shortcut learning

## Next: Step 3 — Preprocessing & augmentation pipeline