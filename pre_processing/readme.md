### Data Pre-processing

1. Origin Data

    Completed.

2. Normalize features

    There are two kinds of normalization. Standardization and Min-max Scaling.

    Completed.<br>
    Please see the funcs of "standardization" and "min_max_scaling" in [processors.py](processors.py) for the details.

3. Over sampling (target == 1)

    referred paper: [http://lin-baobao.com/static/files/papers/SMOTE_Synthetic_Minority_Over_sampling_Technique.pdf](http://lin-baobao.com/static/files/papers/SMOTE_Synthetic_Minority_Over_sampling_Technique.pdf)

4. Under sampling (target == 0)

    Completed.<br>
    Please see the func of "under_sample" in [processors.py](processors.py) for the details.

5. LDA reduce dimensions

    Completed.<br>
    Please see the func of "lda" in [processors.py](processors.py) for the details.

6. combine the above-mentioned methods
