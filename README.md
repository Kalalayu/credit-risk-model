## Proxy Target Variable Engineering (Task 4)

The dataset lacks a direct credit risk label, so we created a proxy target variable `is_high_risk` to identify customers with a higher likelihood of default.

### Steps Performed

1. **RFM Metrics Calculation**  
   - Computed Recency (days since last transaction), Frequency (transaction count), and Monetary (total transaction value) for each `CustomerId`.

2. **Customer Segmentation**  
   - Scaled the RFM features and applied K-Means clustering to divide customers into three groups.  
   - Used a fixed `random_state` to ensure reproducibility.

3. **High-Risk Cluster Identification**  
   - Determined the least engaged cluster (high Recency, low Frequency, low Monetary) as the high-risk segment.

4. **Binary Target Creation**  
   - Assigned `is_high_risk = 1` for customers in the high-risk cluster and `0` for all others.

5. **Integration**  
   - Merged the `is_high_risk` column into the main processed dataset (`train_with_target.csv`) for model training.

### Notes
- `is_high_risk` is a **proxy** based on behavioral patterns and does not reflect actual loan defaults.  
- Using this proxy allows training predictive models for credit risk while acknowledging potential misclassification risks.
