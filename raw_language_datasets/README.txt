
# LANGUAGE IDENTIFICATION DATASET
## CSC423 NLP Term Project

### Dataset Information
- **Created:** 2026-03-15 21:08:14
- **Total Samples:** 1055
- **Languages:** English, Swahili, Lubukusu, Sheng

### File Structure
- `english_raw.csv` - 264 English samples
- `swahili_raw.csv` - 242 Swahili samples
- `lubukusu_raw.csv` - 279 Lubukusu samples
- `sheng_raw.csv` - 270 Sheng samples
- `language_dataset_complete.csv` - Combined dataset (1055 samples)
- `labeled_data.csv` - Labeled dataset for training
- `sample_test_data.csv` - 20 samples per language for testing

### Language Distribution
language
lubukusu    279
sheng       270
english     264
swahili     242

### Dataset Format
All CSV files have the following columns:
- `text`: The raw text sample
- `language`: Language label (english/swahili/lubukusu/sheng)
- `source`: Data source (manual_collection)
- `date_collected`: Collection date

### Usage Notes
1. This is raw data - preprocessing needed before model training
2. Short text samples (1-2 sentences) as per project requirements
3. Balanced dataset across all four languages
4. Sheng includes Kenyan slang and code-mixed expressions

### Next Steps
1. Load the dataset using pandas
2. Apply preprocessing (lowercasing, punctuation removal, etc.)
3. Extract features (TF-IDF, character n-grams)
4. Train classification models
5. Evaluate and compare model performance

### Contact
For questions about this dataset, refer to the course instructor.
