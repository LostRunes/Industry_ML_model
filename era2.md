Temporal Information
it sees:

trends,
dynamics,
evolution,
drift,
process transitions.

Nearly ALL faults became separable.
![alt text](image-2.png)

Process evolution matters more than instantaneous values.
-------------------------
LSTM is not learning at all.
preprocessing mistake.
Your Random Forest worked WITHOUT scaling issues because:

tree models don’t care much about feature scaling.

BUT:

neural networks ABSOLUTELY do.

Right now your features have wildly different range
features have wildly different ranges:

Examples:

some variables near 0,
some near 4000,
some near 100.

The LSTM gradients get destroyed.

So the network collapses into:

predicting one class only


sequence normalization.

![alt text](image-3.png)

Developed a deep-learning-based industrial fault
diagnosis system using the Tennessee Eastman Process benchmark.

Implemented:
- multivariate process monitoring,
- anomaly detection,
- temporal feature engineering,
- LSTM sequence modeling,
- real-time monitoring dashboard.

Improved fault classification accuracy
from 72% (static ML) to 97.6% using
temporal process intelligence.


not added:
https://drive.google.com/file/d/1-iNc8QXjQo29-Az-5pkPmhskMt6J7AzL/view?usp=drive_link 

if not os.path.exists("models/"):
    import download_models
if not os.path.exists("data/"):
    import download_dataset