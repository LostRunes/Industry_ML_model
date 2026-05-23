python train_temporal_model.py  
Dataset Shape: (98000, 211)

Training Temporal Random Forest...


Accuracy:
0.9610714285714286

Classification Report:

              precision    recall  f1-score   support

           1       1.00      1.00      1.00       900
           2       0.99      0.99      0.99       900
           3       0.98      0.99      0.99       900
           4       1.00      0.99      0.99       900
           5       0.94      0.96      0.95      1000
           6       0.96      0.96      0.96      1000
           7       0.96      0.96      0.96      1000
           8       0.95      0.96      0.95      1000
           9       0.97      0.97      0.97      1000
          10       0.94      0.97      0.95      1000
          11       0.97      0.95      0.96      1000
          12       0.94      0.94      0.94      1000
          13       0.96      0.95      0.95      1000
          14       0.96      0.96      0.96      1000
          15       0.98      0.97      0.97      1000
          16       0.97      0.94      0.96      1000
          17       0.95      0.93      0.94      1000
          18       0.94      0.95      0.95      1000
          19       0.96      0.97      0.96      1000
          20       0.92      0.93      0.92      1000

    accuracy                           0.96     19600
   macro avg       0.96      0.96      0.96     19600
weighted avg       0.96      0.96      0.96     19600


Confusion Matrix:

[[897   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   3]
 [  0 887  12   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0]
 [  0   6 893   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0]
 [  0   1   8 891   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0]
 [  0   0   0   0 959   1   4   5   2   5   1   5   3   3   1   2   3   1
    0   5]
 [  0   0   0   0   4 956   1   5   1   5   2   2   2   5   2   0   3   4
    2   6]
 [  0   0   0   0   5   3 960   5   0   4   0   3   2   2   2   0   2   4
    3   5]
 [  0   0   0   0   4   2   4 958   4   4   2   2   1   5   2   4   1   2
    0   5]
 [  0   0   0   0   1   3   3   1 970   7   2   2   0   1   0   1   3   1
    1   4]
 [  0   0   0   0   3   2   2   2   5 966   2   2   2   3   0   1   3   3
    1   3]
 [  0   0   0   0   6   2   3   1   1   4 955   2   5   1   1   4   3   4
    3   5]
 [  0   0   0   0   4   1   2   5   4   3   4 942  13   2   1   1   4   8
    2   4]
 [  0   0   0   0   4   3   1   0   2   4   3  19 947   2   2   3   1   2
    1   6]
 [  0   0   0   0   4   3   0   4   3   4   3   3   1 960   2   1   2   4
    1   5]
 [  0   0   0   0   3   2   2   3   0   5   1   1   0   2 972   3   0   1
    1   4]
 [  0   0   0   0   4   2   5   5   2   2   1   3   4   1   4 944   8   3
    2  10]
 [  0   0   0   0   6   6   3   9   3   2   4   3   2   3   1   5 930  17
    2   4]
 [  0   0   0   0   6   1   3   5   5   2   3   2   1   3   2   0   9 953
    1   4]
 [  0   0   0   0   2   2   1   1   2   1   0   3   3   2   2   0   1   3
  967  10]
 [  0   0   0   0   5   7   4   2   1   6   3   3   1   1   2   3   5   3
   24 930]]

   --------------------------------------------------------------

*lstm* 
    python train_lstm_model.py
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1779557503.911159     464 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
C:\Users\KIIT\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.0) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1779557508.991022     464 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Original X Shape: (95000, 10, 52)
Scaled X Shape: (95000, 10, 52)
I0000 00:00:1779557511.961341     464 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.

Training LSTM...

Epoch 1/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 14s 24ms/step - accuracy: 0.7075 - loss: 0.8852 - val_accuracy: 0.9589 - val_loss: 0.1419
Epoch 2/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 10s 22ms/step - accuracy: 0.9565 - loss: 0.1568 - val_accuracy: 0.9686 - val_loss: 0.1000
Epoch 3/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 11s 22ms/step - accuracy: 0.9664 - loss: 0.1153 - val_accuracy: 0.9698 - val_loss: 0.0952
Epoch 4/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 11s 24ms/step - accuracy: 0.9695 - loss: 0.1002 - val_accuracy: 0.9705 - val_loss: 0.0889
Epoch 5/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 12s 26ms/step - accuracy: 0.9712 - loss: 0.0925 - val_accuracy: 0.9711 - val_loss: 0.0883
Epoch 6/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 12s 25ms/step - accuracy: 0.9726 - loss: 0.0860 - val_accuracy: 0.9714 - val_loss: 0.0862
Epoch 7/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 12s 25ms/step - accuracy: 0.9727 - loss: 0.0870 - val_accuracy: 0.9715 - val_loss: 0.0840
Epoch 8/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 12s 26ms/step - accuracy: 0.9746 - loss: 0.0787 - val_accuracy: 0.9723 - val_loss: 0.0827
Epoch 9/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 11s 24ms/step - accuracy: 0.9742 - loss: 0.0779 - val_accuracy: 0.9714 - val_loss: 0.0822
Epoch 10/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 12s 26ms/step - accuracy: 0.9744 - loss: 0.0768 - val_accuracy: 0.9726 - val_loss: 0.0789
Epoch 11/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 12s 25ms/step - accuracy: 0.9747 - loss: 0.0748 - val_accuracy: 0.9732 - val_loss: 0.0782
Epoch 12/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 12s 25ms/step - accuracy: 0.9749 - loss: 0.0737 - val_accuracy: 0.9733 - val_loss: 0.0765
Epoch 13/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 11s 24ms/step - accuracy: 0.9756 - loss: 0.0713 - val_accuracy: 0.9732 - val_loss: 0.0743
Epoch 14/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 12s 25ms/step - accuracy: 0.9755 - loss: 0.0705 - val_accuracy: 0.9732 - val_loss: 0.0729
Epoch 15/15
475/475 ━━━━━━━━━━━━━━━━━━━━ 12s 25ms/step - accuracy: 0.9752 - loss: 0.0716 - val_accuracy: 0.9736 - val_loss: 0.0727
594/594 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.9766 - loss: 0.0661      

Test Accuracy:
0.976578950881958
594/594 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step    

Classification Report:

              precision    recall  f1-score   support

           0       1.00      0.95      0.97       900
           1       1.00      1.00      1.00       900
           2       1.00      1.00      1.00       900
           3       1.00      1.00      1.00       900
           4       1.00      0.99      0.99       900
           5       1.00      1.00      1.00       900
           6       1.00      0.99      1.00       900
           7       1.00      1.00      1.00       900
           8       1.00      1.00      1.00       900
           9       1.00      0.99      0.99       900
          10       0.98      0.96      0.97      1000
          11       0.73      1.00      0.84      1000
          12       1.00      0.96      0.98      1000
          13       1.00      0.96      0.98      1000
          14       1.00      0.95      0.98      1000
          15       1.00      0.96      0.98      1000
          16       0.98      0.97      0.97      1000
          17       1.00      0.96      0.98      1000
          18       0.99      0.96      0.98      1000
          19       1.00      0.96      0.98      1000

    accuracy                           0.98     19000
   macro avg       0.98      0.98      0.98     19000
weighted avg       0.98      0.98      0.98     19000


LSTM model saved!
PS C:\Users\KIIT\OneDrive\Desktop\industry-model>