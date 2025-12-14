import json
import os

nb_path = r'c:\Users\Jose\Desktop\KAIM8\WEEK4\Codes\Credit Risk Probability Model\credit-risk-model\notebooks\eda.ipynb'

insights_content = [
    "## 9. Top 3-5 Key Insights\n",
    "\n",
    "Based on the exploratory analysis, the following key insights inform our credit risk modeling approach:\n",
    "\n",
    "1.  **Recency as a Dominant Risk Signal**: Customers with long gaps between transactions (High Recency) show significantly lower engagement across all product categories. This \"dormancy\" is a strong proxy for potential churn or default risk, justifying its use as a primary component in our Proxy Default definition.\n",
    "2.  **Transaction Volume & Credit Limits**: There is a clear segmentation in `Amount`. `financial_services` transactions have a higher average value but greater variance compared to low-value, high-frequency `airtime` purchases. This suggests that credit limits should be dynamic based on the `ProductCategory` usage history.\n",
    "3.  **Fraud Implication**: While confirmed fraud cases (`FraudResult=1`) are rare (~0.2%), they represent a definitive \"Bad\" outcome. These instances must be hard-coded as `Default=1` in our training data, overriding any other RFM metrics.\n",
    "4.  **Behavioral Consistency**: A subset of users exhibits high frequency but consistently low monetary value. These \"micro-users\" may be creditworthy for small amounts but risky for larger loans. Our model should differentiate between \"Low Value/Low Risk\" and \"Low Value/High Risk\" (dormant) users."
]

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the cell to update
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            source = cell.get('source', [])
            # Check if it's the target cell (checking first line)
            if source and "## 9. Top 3-5 Key Insights" in source[0]:
                cell['source'] = insights_content
                print("Updated insights cell.")
                break
    else:
        # If not found, append it (though we saw it exists)
        print("Target cell not found, appending...")
        nb['cells'].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": insights_content
        })

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook saved successfully.")

except Exception as e:
    print(f"Error updating notebook: {e}")
