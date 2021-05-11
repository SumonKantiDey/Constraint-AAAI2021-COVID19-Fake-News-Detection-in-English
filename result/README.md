
# Evaluation based on validation data

## Bert Base Uncased (max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5)
```
Accuracy:: 0.9593457943925233
Mcc Score:: 0.918645158470076
Precision:: 0.959518000941115
Recall:: 0.9593457943925233
F_score:: 0.9593205974425014
classification_report::               
                  precision    recall  f1-score   support
         0.0       0.95      0.97      0.96      1120
         1.0       0.97      0.95      0.96      1020

    accuracy                           0.96      2140
   macro avg       0.96      0.96      0.96      2140
weighted avg       0.96      0.96      0.96      2140
```
## RobertaBase (max_len=128, train_batch=16, test_batch=32, epochs = 4, lr = 3e-5)
```
Accuracy:: 0.9640186915887851
Mcc Score:: 0.9280920953519437
Precision:: 0.9642730394692955
Recall:: 0.9640186915887851
F_score:: 0.9639916551714304
classification_report::               
               precision    recall  f1-score   support
         0.0       0.95      0.98      0.97      1120
         1.0       0.97      0.95      0.96      1020

    accuracy                           0.96      2140
   macro avg       0.96      0.96      0.96      2140
weighted avg       0.96      0.96      0.96      2140

```
## RobertaLarge[Last Four Hidden] (max_len=128, train_batch=16, test_batch=32, epochs = 4, lr = 2e-5)
```
Mcc Score:: 0.9419583178172792
Accuracy:: 0.9710280373831776
Precision:: 0.9710722463087809
Recall:: 0.9710280373831776
F_score:: 0.9710195114711334
classification_report:: 
              precision    recall  f1-score   support

         0.0       0.97      0.98      0.97      1120
         1.0       0.98      0.96      0.97      1020

    accuracy                           0.97      2140
   macro avg       0.97      0.97      0.97      2140
weighted avg       0.97      0.97      0.97      2140

```
## oubiobert(max_len=128, train_batch=16, test_batch=32, epochs = 8, lr=3e-5)
```
Mcc Score:: 0.9101072473911911
Accuracy:: 0.9551401869158879
Precision:: 0.9551908578527679
Recall:: 0.9551401869158879
F_score:: 0.9551245086872342
classification_report::               
                  precision    recall  f1-score   support
         0.0       0.95      0.96      0.96      1120
         1.0       0.96      0.95      0.95      1020

    accuracy                           0.96      2140
   macro avg       0.96      0.95      0.96      2140
weighted avg       0.96      0.96      0.96      2140
```
## Covid Bert-v2(max_len=128, train_batch=16, test_batch=32, epochs = 4, lr=3e-5)
```
Mcc Score:: 0.9447627204469508
Accuracy:: 0.9724299065420561
Precision:: 0.9724667109936711
Recall:: 0.9724299065420561
F_score:: 0.9724225360520745
classification_report::               
                  precision    recall  f1-score   support
         0.0       0.97      0.98      0.97      1120
         1.0       0.98      0.97      0.97      1020

    accuracy                           0.97      2140
   macro avg       0.97      0.97      0.97      2140
weighted avg       0.97      0.97      0.97      2140
```
## Sci Bert(max_len=128, train_batch=32, test_batch=32, epochs = 4, lr=3e-5)
```
Mcc Score:: 0.9120956612278952
Accuracy:: 0.9560747663551402
Precision:: 0.9562590601606478
Recall:: 0.9560747663551402
F_score:: 0.9560461258906654
classification_report::               
                  precision    recall  f1-score   support
         0.0       0.95      0.97      0.96      1120
         1.0       0.96      0.94      0.95      1020

    accuracy                           0.96      2140
   macro avg       0.96      0.96      0.96      2140
weighted avg       0.96      0.96      0.96      2140
```

## Bert large Uncased(max_len=128, train_batch=16, test_batch=32, epochs = 5, lr=3e-5)
```
Accuracy:: 0.9677570093457943
Mcc Score:: 0.9353690148654211
Precision:: 0.9677601911104973
Recall:: 0.9677570093457943
F_score:: 0.9677533041269694
classification_report::               
                  precision    recall  f1-score   support
         0.0       0.97      0.97      0.97      1120
         1.0       0.97      0.96      0.97      1020

    accuracy                           0.97      2140
   macro avg       0.97      0.97      0.97      2140
weighted avg       0.97      0.97      0.97      2140
```

## LSTMBase(Glove300, batch=32, epochs = 15, lr = 0.001, weight_decay = 1e-4)
```
Accuracy:: 0.8551401869158879
Precision:: 0.8551541083212281
Recall:: 0.8551401869158879
F_score:: 0.8551463997273135
classification_report::               
               precision    recall  f1-score   support
         0.0       0.86      0.86      0.86      1120
         1.0       0.85      0.85      0.85      1020

    accuracy                           0.86      2140
   macro avg       0.85      0.85      0.85      2140
weighted avg       0.86      0.86      0.86      2140
```
## LSTM Attention (Glove300, batch=32, epochs = 15, lr = 0.001, weight_decay = 1e-4)
```
Accuracy:: 0.9175644028103045
Precision:: 0.9194067835563549
Recall:: 0.9175644028103045
F_score:: 0.9176085797007608
classification_report::               
               precision    recall  f1-score   support
         0.0       0.95      0.89      0.92      1119
         1.0       0.89      0.95      0.92      1016

    accuracy                           0.92      2135
   macro avg       0.92      0.92      0.92      2135
weighted avg       0.92      0.92      0.92      2135
```