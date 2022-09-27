# Logistic Regression with Gradient Ascent

Python script implementing a Logistic Regression machine learning algorithm for binary input/output data utilizing gradient ascent to develop values to predict future outcomes based on provided new data.

## Implementation

This code requires a "training file" which needs to be in the format of lists/rows of binary data. The last value in the row represents the overall outcome/label of all previous entries/paramaters in the respective row. Once the training is complete, a testing file can be used to determine the level of precision in which the code can predict future outcomes based on given new data. 

## Example Data

` index        Date   Time                     Location  ... Aboard Fatalities Ground                                            Summary
1948   1948  12/11/1964    NaN  Near Da Nang, South Vietnam  ...   38.0       38.0    0.0  Hit a mountain shortly after taking off explod...
831     831  12/24/1946  19:19   Near San Diego, California  ...   12.0       12.0    0.0  Crashed east slope of Cuyapaipe Mountain at 6,...
3840   3840  05/05/1989    NaN          Near Cancun, Mexico  ...   19.0        6.0    0.0  Crashed after the pilot radioed he was making ...
1877   1877  10/20/1963    NaN         Hayes Island, Russia  ...    5.0        5.0    0.0                                                NaN
1991   1991  07/25/1965  07:40    Near Libacao, Philippines  ...   37.0       37.0    0.0  Crashed and burned on Penay Island on a domest... `

## Notes

* Coded in Python