## Proxy Target Variable Engineering (Task 4)

Since the dataset does not include a direct credit risk label, we created a proxy target variable `is_high_risk` to indicate customers with a higher likelihood of default. 

### Steps Taken

1. **RFM Metrics Calculation**  
   - For each `CustomerId`, we calculated Recency (days since last transaction), Frequency (number of transactions), and Monetary (total transaction value) based on transaction history.

2. **Customer Segmentation**  
   - Scaled the RFM features and applied K-Means clustering to group customers into three segments.  
   - Set a fixed `random_state` for reproducibility.

3. **High-Risk Cluster Identification**  
   - Identified the least engaged customer cluster (highest Recency, lowest Frequency and Monetary) as the high-risk segment.

4. **Binary Target Creation**  
   - Assigned `is_high_risk = 1` to customers in the high-risk cluster and `0` to others.

5. **Integration**  
   - Merged `is_high_risk` back into the main processed dataset (`train_with_target.csv`) for model training.

### Notes
- This target is a **proxy** based on behavioral patterns and does not represent actual loan defaults.  
- The proxy allows training predictive models for credit risk while highlighting potential business risks of misclassification.
