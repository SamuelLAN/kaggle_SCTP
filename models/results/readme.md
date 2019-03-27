# Directory for Save Results

Model:

    LightGBM

Param:

    max_depth=-1, n_estimators=999999, learning_rate=0.02, colsample_bytree=0.3, num_leaves=2, objective='binary', boosting_type='goss'

AUC for different dataset:

    origin: 0.905002(test)
    origin_min_max_scaling: 0.905267(test)
    origin_standardization_min_max_scaling: 0.905261(test) 0.898(real test)
    under_sample_3.0: 0.897255(val), 0.904868(test)
    under_sample_3.0_min_max_scaling:
    under_sample_5.0:
    under_sample_5.0_standardization_min_max_scaling: 0.896987(val), 0.904629(test)
    add_lda: 0.895826(val),0.904800(test)
    add_lda_standardization_min_max_scaling: 0.895608(val),0.905010(test), 0.898(real test)

    origin_1:
    origin_1_standardization_min_max_scaling: 0.90123(val),0.902043(test)
    add_lda_1: 0.900274(val), 0.900367(test)
    add_lda_1_standardization_min_max_scaling: 0.900274(val), 0.900367(test)

    ---

    origin_7_min_max_scaling_aug_3.0: 0.9029 (test)


----

Model:

    CatBoost

Param:

    max_depth=2, learning_rate=0.02, colsample_bylevel=0.03, objective="Logloss",

AUC for different dataset:

    origin:
    origin_min_max_scaling: 0.905826
    add_lda_standardization_min_max_scaling:
    origin_1_standardization_min_max_scaling: 0.902350

----

Model:

    XgBoost

Param:

    max_depth=2, colsample_bytree=0.3, learning_rate=0.02, objective='binary:logistic',

AUC for different dataset:

    origin:
    origin_min_max_scaling: 0.902554(test)
    add_lda_standardization_min_max_scaling:

----

Model:

    Shadow Neural Networks

AUC for different dataset:

    origin_min_max_scaling: 0.861923(val), 0.870252(test)


