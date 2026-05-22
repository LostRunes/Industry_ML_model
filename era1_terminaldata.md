Train Accuracy: 0.2750
Test Accuracy:  0.2250
Test Macro F1:  0.1983

Test Classification Report:
              precision    recall  f1-score   support

           0       0.79      0.19      0.30       145
           1       0.12      0.33      0.17        18
           2       0.12      0.32      0.17        19
           3       0.09      0.33      0.15        18

    accuracy                           0.23       200
   macro avg       0.28      0.29      0.20       200
weighted avg       0.61      0.23      0.26       200


============================================================      
         MODEL SUMMARY & RECOMMENDATIONS
============================================================      
Saving the best model: Regularized Random Forest
Model successfully saved to 'models/fault_classifier.pkl'!

ACTIONABLE ADVICE TO INCREASE ACCURACY:
 1. Fix the Data Generation Pipeline: Verify that the rows of features were not shuffled
    or generated using pure random noise independent of the 'Fault_Type' label.
 2. Check the FFT extraction: The columns 'FFT_Temp_X' etc. do not correspond to the actual
    Fourier transform of the 'Temperature' columns. Make sure the     frequency aggregation are correctly aligned with row indices. 
 3. Collect Real Time-Series Data: If you have access to the raw sequential sensor measurements,
    we can compute actual sequential/temporal features (lags, rolling averages, rolling variance)
    or run temporal deep learning architectures (LSTMs or 1D-CNNs).
============================================================      

(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> pip install pyreadr
Collecting pyreadr
  Downloading pyreadr-0.5.6-cp313-cp313-win_amd64.whl.metadata (1.4 kB)
Requirement already satisfied: pandas>=1.2.0 in c:\users\kiit\onedrive\desktop\industry-model\venv\lib\site-packages (from pyreadr) (3.0.3)
Requirement already satisfied: numpy>=1.26.0 in c:\users\kiit\onedrive\desktop\industry-model\venv\lib\site-packages (from pandas>=1.2.0->pyreadr) (2.4.6)
Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\kiit\onedrive\desktop\industry-model\venv\lib\site-packages (from pandas>=1.2.0->pyreadr) (2.9.0.post0)
Requirement already satisfied: tzdata in c:\users\kiit\onedrive\desktop\industry-model\venv\lib\site-packages (from pandas>=1.2.0->pyreadr) (2026.2)
Requirement already satisfied: six>=1.5 in c:\users\kiit\onedrive\desktop\industry-model\venv\lib\site-packages (from python-dateutil>=2.8.2->pandas>=1.2.0->pyreadr) (1.17.0)
Downloading pyreadr-0.5.6-cp313-cp313-win_amd64.whl (2.4 MB)      
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.4/2.4 MB 29.0 MB/s  0:00:00
Installing collected packages: pyreadr
Successfully installed pyreadr-0.5.6

[notice] A new release of pip is available: 25.2 -> 26.1.1        
[notice] To update, run: python.exe -m pip install --upgrade pip  
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model>
 *  History restored 








PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python explore_tep.py
Traceback (most recent call last):
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\explore_tep.py", line 1, in <module>
    import pyreadr
ModuleNotFoundError: No module named 'pyreadr'
PS C:\Users\KIIT\OneDrive\Desktop\industry-model> venv/Scripts/activate
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python explore_tep.py
Traceback (most recent call last):
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\explore_tep.py", line 4, in <module>
    result = pyreadr.read_r("TEP_FaultFree_Training.RData")
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\pyreadr\pyreadr.py", line 67, in read_r
    raise PyreadrError("File {0} does not exist!".format(filename_bytes))
pyreadr.custom_errors.PyreadrError: File b'TEP_FaultFree_Training.RData' does not exist!
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> 
 *  History restored 

PS C:\Users\KIIT\OneDrive\Desktop\industry-model> venv/Scripts/activate
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python explore_tep.py

Objects inside file:
odict_keys(['fault_free_training'])

First 5 rows:
   faultNumber  simulationRun  sample  ...   xmv_9  xmv_10  xmv_11
0          0.0            1.0       1  ...  47.510  41.258  18.447
1          0.0            1.0       2  ...  47.427  41.359  17.194
2          0.0            1.0       3  ...  47.468  41.199  20.530
3          0.0            1.0       4  ...  47.658  41.643  18.089
4          0.0            1.0       5  ...  47.346  41.507  18.461

[5 rows x 55 columns]

Shape:
(250000, 55)

Columns:
Index(['faultNumber', 'simulationRun', 'sample', 'xmeas_1', 'xmeas_2',
       'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8',
Index(['faultNumber', 'simulationRun', 'sample', 'xmeas_1', 'xmeas_2',
       'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8',
       'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14',
       'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20',
       'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26',
       'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32',
       'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38',
       'xmeas_39', 'xmeas_40', 'xmeas_41', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4',
       'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmvIndex(['faultNumber', 'simulationRun', 'sample', 'xmeas_1', 'xmeas_2',
       'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8',
       'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14',
       'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19',       'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8',
as_8',
       'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14',
       'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20',
       'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26',
       'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32',
       'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38',
       'xmeas_39', 'xmeas_40', 'xmeas_41', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4',
       'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11'],
      dtype='str')                                       python inspect_faults.pyers\KIIT\OneDrive\Desktop\industry-model>
Traceback (most recent call last):
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\inspect_faults.py", line 5, in <module>
    result = pyreadr.read_r("TEP_Faulty_Training.RData")
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\pyreadr\pyreadr.py", line 67, in read_r
    raise PyreadrError("File {0} does not exist!".format(filename_bytes))
pyreadr.custom_errors.PyreadrError: File b'TEP_Faulty_Training.RData' does not exist!
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python inspect_faults.py       

Dataset Shape:
(5000000, 55)

Fault Counts:
faultNumber
1     250000
2     250000
3     250000
4     250000
5     250000
6     250000
7     250000
8     250000
9     250000
10    250000
11    250000
12    250000
13    250000
14    250000
15    250000
16    250000
17    250000
18    250000
19    250000
20    250000
Name: count, dtype: int64
6     250000
7     250000
8     250000
9     250000
10    250000
11    250000
12    250000
13    250000
14    250000
15    250000
16    250000
17    250000
18    250000
19    250000
20    250000
Name: count, dtype: int64
9     250000
10    250000
11    250000
12    250000
13    250000
14    250000
15    250000
16    250000
17    250000
18    250000
19    250000
20    250000
Name: count, dtype: int64
13    250000
14    250000
15    250000
16    250000
17    250000
18    250000
19    250000
20    250000
Name: count, dtype: int64
15    250000
16    250000
17    250000
18    250000
19    250000
20    250000
Name: count, dtype: int64
17    250000
18    250000
19    250000
20    250000
Name: count, dtype: int64
20    250000
Name: count, dtype: int64

Unique Faults:
[np.int32(1), np.int32(2), np.int32(3), np.int32(4), np.int32(5), np.int32(6), np.int32(7), np.int32(8), np.int32(9), np.int32(10), np.int32(11), np.int32(12), np.int32(13), np.int32(14), np.int32(15), np.int32(16), np.int32(17), np.int32(18), np.int32(19), np.int32(20)]
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python create_subset.py
Original Shape: (5000000, 55)
Subset Shape: (100000, 54)
Subset saved!
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model>
Unique Faults:
[np.int32(1), np.int32(2), np.int32(3), np.int32(4), np.int32(5), np.int32(6), np.int32(7), np.int32(8), np.int32(9), np.int32(10), np.int32(11), np.int32(12), np.int32(13), np.int32(14), np.int32(15), np.int32(16), np.int32(17), np.int32(18), np.int32(19), np.int32(20)]
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python create_subset.py
Unique Faults:
[np.int32(1), np.int32(2), np.int32(3), np.int32(4), np.int32(5), np.int32(6), np.int32(7), np.int32(8), np.int32(9), np.int32(10), np.int32(11), np.int32(12), np.int32(13), np.int32(14), np.int32(15), np.int32(16), np.int32(17), np.int32(18), np.int32(19), np.intUnique Faults:
Unique Faults:
[np.int32(1), np.int32(2), np.int32(3), np.int32(4), np.int32(5), np.int32(6), np.int32(7), np.int32(8), np.int32(9), np.int32(10), np.int32(11), np.int32(12), np.int32(13), np.int32(14), np.int32(15), np.int32(16), np.int32(17), np.int32(18), np.int32(19), np.int32(20)]
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python create_subset.py
Original Shape: (5000000, 55)
Subset Shape: (100000, 54)
Subset saved!                                            python create_subset.pysers\KIIT\OneDrive\Desktop\industry-model>
Original Shape: (5000000, 55)
Subset Shape: (100000, 54)
Subset saved!
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python create_subset.py
Traceback (most recent call last):
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\create_subset.py", line 5, in <module>
    result = pyreadr.read_r("TEP_Faulty_Training.RData")
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\pyreadr\pyreadr.py", line 67, in read_r
    raise PyreadrError("File {0} does not exist!".format(filename_bytes))
pyreadr.custom_errors.PyreadrError: File b'TEP_Faulty_Training.RData' does not exist!
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python create_subset.py
odict_keys(['faulty_training'])
Original Shape: (5000000, 55)
Subset Shape: (100000, 55)

Columns:
Index(['faultNumber', 'simulationRun', 'sample', 'xmeas_1', 'xmeas_2',
       'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8',
       'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14',
       'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20',
       'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26',
       'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32',
       'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38',
       'xmeas_39', 'xmeas_40', 'xmeas_41', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4',
       'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11'],
      dtype='str')

Fault Counts:
faultNumber
1     5000
2     5000
3     5000
4     5000
5     5000
6     5000
7     5000
8     5000
9     5000
10    5000
11    5000
12    5000
13    5000
14    5000
15    5000
16    5000
17    5000
18    5000
19    5000
20    5000
Name: count, dtype: int64

Subset saved successfully!
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model>









                                                              
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python pca_visualizer.py
C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Scripts\python.exe: can't open file 'C:\\Users\\KIIT\\OneDrive\\Desktop\\industry-model\\pca_visualizer.py': [Errno 2] No such file or directory   
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python pca_visualization.py
Traceback (most recent call last):
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\pca_visualization.py", line 52, in <module>
    plt.show()
    ~~~~~~~~^^
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\matplotlib\pyplot.py", line 613, in show
    return _get_backend_mod().show(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\matplotlib\backend_bases.py", line 3550, in show       
    cls.mainloop()
    ~~~~~~~~~~~~^^
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\matplotlib\backends\_backend_tk.py", line 572, in start_main_loop
    first_manager.window.mainloop()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\KIIT\AppData\Local\Programs\Python\Python313\Lib\tkinter\__init__.py", line 1599, in mainloop
    self.tk.mainloop(n)
    ~~~~~~~~~~~~~~~~^^^
KeyboardInterrupt
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python train_tep_model.py

Accuracy:
0.72035

Classification Report:
              precision    recall  f1-score   support

           1       0.96      0.95      0.96      1000
           2       0.95      0.94      0.95      1000
           3       0.01      0.01      0.01      1000
           4       0.88      0.92      0.90      1000
           5       0.92      0.91      0.91      1000
           6       0.94      0.97      0.96      1000
           7       0.95      0.96      0.96      1000
           8       0.90      0.88      0.89      1000
           9       0.01      0.01      0.01      1000
          10       0.72      0.75      0.74      1000
          11       0.81      0.76      0.78      1000
          12       0.82      0.86      0.84      1000
          13       0.86      0.79      0.82      1000
          14       0.95      0.91      0.93      1000
          15       0.02      0.01      0.01      1000
          16       0.87      0.70      0.78      1000
          17       0.80      0.84      0.82      1000
          18       0.78      0.78      0.78      1000
          19       0.88      0.78      0.82      1000
          20       0.75      0.68      0.71      1000

    accuracy                           0.72     20000
   macro avg       0.74      0.72      0.73     20000
weighted avg       0.74      0.72      0.73     20000


Confusion Matrix:
[[954   7   1   2   1   2   3   3   5   0   2   3   2   1   0   2   1   1
    3   7]
 [  1 942   7   5   2   3   1   3   3   1   4   3   6   1   8   2   3   3
    1   1]
 [  0   1  11   3   4   3   0   5 565   7  20   3  12   1 277   8  14  33
    9  24]
 [  1   2   2 921   3   3   3   3   3   1  39   2   3   0   1   2   3   3
    2   3]
 [  1   3   5   5 907   3   4   1   8  16   4  10   1   3  10   3   3   3
    5   5]
 [  3   1   2   0   2 967   1   2   2   0   2   2   3   2   1   2   2   3
    1   2]
 [  2   4   0   5   1   4 957   2   7   3   2   1   0   3   3   1   1   2
    0   2]
 [ 12   3   4   1   4   2   2 882   9  10   4  25  10   3   3   4   8   2
    4   8]
 [  2   4 611   6   6   2   1   3  13  14  11   1   9   2 222   3  24  32
    9  25]
 [  1   3  35   1  10   2   4   8  30 747   5   6  16   3  34  32  20   7
   15  21]
 [  3   5  70  65   1   3   1   4  23   6 756   5   2  13  21   3   4   6
    5   4]
 [  0   1   3   5  16   5   1  15   7  16   9 859   6   2  10   3   2  31
    2   7]
 [  0   2  16   4   1   6   0  19  19  25   4  36 788   2  16   1  20  18
    2  21]
 [  0   5   1   3   3   3   5   3   8   2   6   2   2 910   3   2  39   0
    1   2]
 [  2   3 486   4   8   2   3   3 370   9  12   6  12   1  13   8   5  15
   15  23]
 [  1   0  36   3   3   2   3   6  40 108   9   3   9   2  36 704   5   4
   12  14]
 [  2   3  28   5   4   4   3   5  11   8  11   4  12   6   7   5 839  25
    0  18]
 [  2   1  17   4   2   5   3   5  18   8   4  72  12   0   7   3  [  1   0  36   3   3   2   3   6  40 108   9   3   9   2  36 704   5   4
   12  14]
 [  2   3  28   5   4   4   3   5  11   8  11   4  12   6   7   5 839  25
    0  18]
 [  2   1  17   4   2   5   3   5  18   8   4  72  12   0   7   3  34 780
    2  21]
 [  4   0  62   4   6   2   5   1  34   7  27   1   4   0  29  12   4   4
  776  18]
 [  4   1  50   6   4   1   3  10  41  43   4   4  12   2  55  12  21  25
   21 681]]
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model>








 [  1   0  36   3   3   2   3   6  40 108   9   3   9   2  36 704   5   4
   12  14]
 [  2   3  28   5   4   4   3   5  11   8  11   4  12   6   7   5 839  25
    0  18]
 [  2   1  17   4   2   5   3   5  18   8   4  72  12   0   7   3  34 780
 [  2   3  28   5   4   4   3   5  11   8  11   4  12   6   7   5 839  25
    0  18]
 [  2   1  17   4   2   5   3   5  18   8   4  72  12   0   7   3     0  18]
 [  2   1  17   4   2   5   3   5  18   8   4  72  12   0   7   3  [  2   1  17   4   2   5   3   5  18   8   4  72  12   0   7   3  34 780
    2  21]
 [  4   0  62   4   6   2   5   1  34   7  27   1   4   0  29  12   4   4
  776  18]
 [  4   1  50   6   4   1   3  10  41  43   4   4  12   2  55  12  21  25
   21 681]]
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python feature_importance.py

Top 15 Most Important Features:

     Feature  Importance
50    xmv_10    0.076353
20  xmeas_21    0.057128
44     xmv_4    0.054508
8    xmeas_9    0.045367
0    xmeas_1    0.044806
43     xmv_3    0.034166
18  xmeas_19    0.033584
49     xmv_9    0.029414
17  xmeas_18    0.028067
21  xmeas_22    0.027563
45     xmv_5    0.027135
9   xmeas_10    0.026780
51    xmv_11    0.026329
46     xmv_6    0.025080
10  xmeas_11    0.021528
Traceback (most recent call last):
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\feature_importance.py", line 77, in <module>
    plt.show()
    ~~~~~~~~^^
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\matplotlib\pyplot.py", line 613, in show
    return _get_backend_mod().show(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\matplotlib\backend_bases.py", line 3550, in show       
    cls.mainloop()
    ~~~~~~~~~~~~^^
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\matplotlib\backends\_backend_tk.py", line 572, in start_main_loop
    first_manager.window.mainloop()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\KIIT\AppData\Local\Programs\Python\Python313\Lib\tkinter\__init__.py", line 1599, in mainloop
    self.tk.mainloop(n)
    ~~~~~~~~~~~~~~~~^^^
KeyboardInterrupt
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python train_anomaly_detector.py
Traceback (most recent call last):
e-packages\matplotlib\backend_bases.py", line 3550, in show       
    cls.mainloop()
    ~~~~~~~~~~~~^^
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\matplotlib\backends\_backend_tk.py", line 572, in start_main_loop
    first_manager.window.mainloop()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\KIIT\AppData\Local\Programs\Python\Python313\Lib\tkinter\__init__.py", line 1599, in mainloop
    self.tk.mainloop(n)
    ~~~~~~~~~~~~~~~~^^^
KeyboardInterrupt
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python train_anomaly_detector.py
Traceback (most recent call last):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\KIIT\AppData\Local\Programs\Python\Python313\Lib\tkinter\__init__.py", line 1599, in mainloop
    self.tk.mainloop(n)
    ~~~~~~~~~~~~~~~~^^^
KeyboardInterrupt
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python train_anomaly_detector.py
Traceback (most recent call last):
    self.tk.mainloop(n)
    ~~~~~~~~~~~~~~~~^^^
KeyboardInterrupt
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python train_anomaly_detector.py
Traceback (most recent call last):
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python train_anomaly_detector.py
Traceback (most recent call last):
ain_anomaly_detector.py
Traceback (most recent call last):
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\train_anomaly_detector.py", line 12, in <module>
y_detector.py", line 12, in <module>
    normal_result = pyreadr.read_r(
    normal_result = pyreadr.read_r(
        "TEP_FaultFree_Training.RData"
        "TEP_FaultFree_Training.RData"
    )
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\pyreadr\pyreadr.py", line 67, in read_r
    raise PyreadrError("File {0} does not exist!".format(filename_bytes))
pyreadr.custom_errors.PyreadrError: File b'TEP_FaultFree_Training.RData' does not exist!                                            
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> 
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python train_anomaly_detector.py

Fault-Free Training Shape:
(250000, 55)
Traceback (most recent call last):
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\train_anomaly_detector.py", line 25, in <module>
    faulty_result = pyreadr.read_r(
        "TEP_Faulty_Testing.RData"
    )
  File "C:\Users\KIIT\OneDrive\Desktop\industry-model\venv\Lib\site-packages\pyreadr\pyreadr.py", line 67, in read_r
    raise PyreadrError("File {0} does not exist!".format(filename_bytes))
pyreadr.custom_errors.PyreadrError: File b'TEP_Faulty_Testing.RData' does not exist!
(venv) PS C:\Users\KIIT\OneDrive\Desktop\industry-model> python train_anomaly_detector.py

Fault-Free Training Shape:
(250000, 55)

Faulty Testing Shape:
(9600000, 55)

Training Isolation Forest...

Classification Report:

              precision    recall  f1-score   support

      Normal       0.63      0.90      0.74     20000
     Anomaly       0.82      0.46      0.59     20000

    accuracy                           0.68     40000
   macro avg       0.72      0.68      0.66     40000
weighted avg       0.72      0.68      0.66     40000




PS C:\Users\KIIT\OneDrive\Desktop\industry-model> & c:/Users/KIIT/AppData/Local/Microsoft/WindowsApps/python3.13.exe c:/Users/KIIT/OneDrive/Desktop/industry-model/analyze_tep.py

Dataset Shape:
(100000, 55)

Columns:
Index(['faultNumber', 'simulationRun', 'sample', 'xmeas_1', 'xmeas_2',
       'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8',
       'xmeas_9', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14',
       'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20',
       'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26',
       'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32',
       'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38',
       'xmeas_39', 'xmeas_40', 'xmeas_41', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4',
       'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10', 'xmv_11'],
      dtype='object')

Missing Values:
0

Fault Counts:
faultNumber
1     5000
2     5000
3     5000
4     5000
5     5000
6     5000
7     5000
8     5000
9     5000
10    5000
11    5000
12    5000
13    5000
14    5000
15    5000
16    5000
17    5000
18    5000
19    5000
20    5000
Name: count, dtype: int64

Basic Statistics:
        faultNumber  simulationRun  ...         xmv_10         xmv_11
count  100000.00000  100000.000000  ...  100000.000000  100000.000000
mean       10.50000     251.752800  ...      41.979149      18.851982
std         5.76631     144.412467  ...       9.991357       5.172238
min         1.00000       1.000000  ...      -0.242480      -0.005327
25%         5.75000     126.000000  ...      40.575000      17.155000
50%        10.50000     255.000000  ...      41.209500      18.305000
75%        15.25000     377.000000  ...      41.940000      19.544000
max        20.00000     500.000000  ...     100.300000     100.010000

982
std         5.76631     144.412467  ...       9.991357       5.172238
min         1.00000       1.000000  ...      -0.242480      -0.005327
25%         5.75000     126.000000  ...      40.575000      17.155000
982
std         5.76631     144.412467  ...       9.991357       5.172238
min         1.00000       1.000000  ...      -0.242480      -0.005982
std         5.76631     144.412467  ...       9.991357       5.172982
982
std         5.76631     144.412467  ...       9.991357       5.172238
min         1.00000       1.000000  ...      -0.242480      -0.005327
25%         5.75000     126.000000  ...      40.575000      17.155000
50%        10.50000     255.000000  ...      41.209500      18.305000
75%        15.25000     377.000000  ...      41.940000      19.544000
max        20.00000     500.000000  ...     100.300000     100.010000

[8 rows x 55 columns]
PS C:\Users\KIIT\OneDrive\Desktop\industry-model> ^C
PS C:\Users\KIIT\OneDrive\Desktop\industry-model> & c:/Users/KIIT/AppData/Local/Microsoft/WindowsApps/python3.13.exe c:/Users/KIIT/OneDrive/Desktop/industry-model/pca_visualization.py
75%        15.25000     377.000000  ...      41.940000      19.544000
max        20.00000     500.000000  ...     100.300000     100.010000

[8 rows x 55 columns]
PS C:\Users\KIIT\OneDrive\Desktop\industry-model> ^C
PS C:\Users\KIIT\OneDrive\Desktop\industry-model> & c:/Users/KIIT/AppData/Local/Microsoft/WindowsApps/python3.13.exe c:/Users/KIIT/OneDrive/Desktop/industry-model/pca_visualization.py
[8 rows x 55 columns]
PS C:\Users\KIIT\OneDrive\Desktop\industry-model> ^C
PS C:\Users\KIIT\OneDrive\Desktop\industry-model> & c:/Users/KIIT/AppData/Local/Microsoft/WindowsApps/python3.13.exe c:/Users/KIIT/OneDrive/Desktop/industry-model/pca_visualization.py