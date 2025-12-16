# credit-risk-model
Week 4- Credit-Risk-Probability-Model-using-Alternative-Data
# Credit-Risk-Probability-Model-using-Alternative-Data
End-to-end credit risk modeling using alternative eCommerce transaction data. The project builds a proxy default signal from customer behavior, trains probability-of-default models, converts risk into credit scores, and deploys predictions via a FastAPI service with MLOps, testing, and CI/CD.

## Credit Scoring Business Understanding

### Basel II and Model Interpretability

The Basel II Capital Accord places strong emphasis on accurate measurement, monitoring, and management of credit risk, particularly through the estimation of Probability of Default (PD). This regulatory framework requires financial institutions to justify how risk estimates are produced and to ensure that models are transparent, auditable, and well-documented. As a result, our credit scoring model must not only perform well statistically, but also be interpretable and explainable to regulators, internal risk teams, and business stakeholders. Clear feature definitions, documented assumptions, and reproducible training processes are essential to meet these requirements and support responsible credit decision-making.

### Need for a Proxy Default Variable

The dataset provided does not contain an explicit default label, as customers have not previously taken formal loans through the platform. To enable supervised learning, it is therefore necessary to construct a proxy variable that approximates credit risk based on observable behavioral patterns such as Recency, Frequency, and Monetary (RFM) metrics, transaction consistency, and fraud indicators. This proxy allows us to distinguish between high-risk and low-risk customers. However, using a proxy introduces business risk, including potential misclassification and bias, since the proxy may not perfectly represent true loan default behavior. These limitations must be acknowledged, monitored, and revisited as real repayment data becomes available.

### Trade-offs Between Simple and Complex Models

There is an important trade-off between model interpretability and predictive performance in regulated financial environments. Simple models such as Logistic Regression combined with Weight of Evidence (WoE) transformations offer high transparency, stability, and ease of explanation, making them well-suited for regulatory compliance and stakeholder trust. In contrast, more complex models such as Gradient Boosting or Random Forests may achieve higher predictive accuracy by capturing nonlinear relationships, but they are harder to interpret and explain. In practice, financial institutions often favor simpler, well-understood models for core credit decisions, or use complex models as challenger models under strict governance and validation frameworks.