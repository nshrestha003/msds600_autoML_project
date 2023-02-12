import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=25)

# Average CV score on the training set was: 0.8020937713258135
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_classif, alpha=0.045),
    StackingEstimator(estimator=SGDClassifier(alpha=0.01, eta0=0.1, fit_intercept=False, l1_ratio=0.0, learning_rate="invscaling", loss="squared_hinge", penalty="elasticnet", power_t=50.0)),
    PCA(iterated_power=9, svd_solver="randomized"),
    ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.9000000000000001, min_samples_leaf=12, min_samples_split=17, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 25)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
