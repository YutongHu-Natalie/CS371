from sklearn.metrics import recall_score

y_true= [0,0,0,0,2,0,0,3,0,0,0,0,0,4,0]
y_pred_list=[
    [2,0,0,0,2,0,4,2,0,0,0,0,0,4,0],
    [0,0,0,0,2,0,4,3,0,0,0,0,0,4,0],
    [0,0,0,0,3,0,4,3,0,0,0,0,0,4,0],
    [0,0,0,0,2,2,3,2,0,0,0,0,0,4,2]
]
for y_pred in y_pred_list:
    print(recall_score(y_true, y_pred, average='macro', labels=[0,1,2,3,4], zero_division=0))