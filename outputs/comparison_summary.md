# DEEPVISION: MNIST Model Comparison Summary

| Architecture | CV Val Accuracy | CV Val Loss | CV Val Macro F1 | CV Val Precision | CV Val Recall |
|---|---|---|---|---|---|
| BASIC | 97.43% ± 0.16% | 0.0839 ± 0.0039 | 0.9741 ± 0.0016 | 0.9744 ± 0.0015 | 0.9741 ± 0.0016 |
| IMPROVED | 98.99% ± 0.05% | 0.0318 ± 0.0025 | 0.9899 ± 0.0005 | 0.9899 ± 0.0005 | 0.9899 ± 0.0005 |

### Official Untouched Test Set Results (IMPROVED):
- **Test Accuracy**: 99.26%
- **Test Loss**: 0.0222
- **Test Macro F1**: 0.9926
- **Test Macro Precision**: 0.9926
- **Test Macro Recall**: 0.9925

```
              precision    recall  f1-score   support

           0     0.9959    0.9980    0.9969       980
           1     0.9956    0.9947    0.9952      1135
           2     0.9971    0.9922    0.9947      1032
           3     0.9911    0.9921    0.9916      1010
           4     0.9929    0.9949    0.9939       982
           5     0.9899    0.9910    0.9905       892
           6     0.9958    0.9864    0.9911       958
           7     0.9780    0.9961    0.9870      1028
           8     0.9959    0.9887    0.9923       974
           9     0.9940    0.9911    0.9926      1009

    accuracy                         0.9926     10000
   macro avg     0.9926    0.9925    0.9926     10000
weighted avg     0.9926    0.9926    0.9926     10000

```
