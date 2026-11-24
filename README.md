# De-Identification of Medical Records Using Machine Learning

Major-project implementation using:
1. CNN (PyTorch) – token-level PHI classification
2. Decision Tree – feature-based token classification
3. Random Forest – feature-based ensemble of decision trees
4. Proposed Ensemble – weighted voting of CNN + Decision Tree + Random Forest
5. Regex safeguards – structured PHI such as phone, email, dates and MRN

Input:
- PDF
- Excel (.xlsx/.xls)

Output:
- De-identified PDF
- De-identified Excel

IMPORTANT:
- Development dataset is synthetic. Do not upload real patient records.
- The first version produces a clean de-identified PDF from extracted text; it does not yet preserve the exact original PDF layout.
