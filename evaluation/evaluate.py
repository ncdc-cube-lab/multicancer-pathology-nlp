import pandas as pd
from sklearn.metrics import classification_report

# Performance evaluation
results = []

for variable in ttt['variable'].unique():
    print(f"Performance evaluation - {variable}")
    df_filtered = ttt[ttt['variable']==variable]
    y_true = df_filtered["True_value"].tolist()
    y_pred = df_filtered["pred_value"].tolist()
    
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report["weighted avg"]["f1-score"]
    
    results.append([variable, accuracy, precision, recall, f1])
    
results_df = pd.DataFrame(results, columns = ['var', 'Accuracy', 'Precision', 'Recall','F1-Score'])
results_df = results_df.round(4)