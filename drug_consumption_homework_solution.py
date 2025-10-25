# Drug Consumption (Quantified) - Homework solution
# Runnable Python script / notebook
# Requirements: pandas, numpy, scikit-learn, matplotlib

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load data (raw CSV hosted on GitHub mirror)
url = 'https://raw.githubusercontent.com/deepak525/Drug-Consumption/master/drug_consumption.csv'
raw = pd.read_csv(url)

# Quick peek
print('Raw shape:', raw.shape)
print(raw.columns.tolist())

# According to UCI description, columns mapping (the CSV may already contain headers).
# If headers are generic or missing, adjust names accordingly. We assume the CSV has descriptive headers.

# Inspect head
print(raw.head())

# Identify feature columns (12 predictors) and 18 drug target columns.
# Common column names in mirrors: ['ID','Age','Gender','Education','Country','Ethnicity','Nscore','Escore','Oscore','Ascore','Cscore','Impulsivity','Sensation','Alcohol','Amphet','...']

# We'll programmatically detect numeric feature columns that are predictors vs drug columns by name heuristic.
candidate_targets = [c for c in raw.columns if c.lower() in ('alcohol','amphet','amyl','benzos','cannabis','choc','coke','crack','ecstasy','heroin','ketamine','legal highs','lsd','methadone','mushrooms','nicotine','vsa','semeron','amphetamines')]
# But to be robust, we'll identify drugs as columns with <=7 unique values and containing usage labels sometimes encoded as strings in original mirrored csv.

# If drug columns are textual like 'CL0','CL1' etc., the dataset sometimes uses 'CL0..CL6' codes; many mirrors use numeric 0..6 mapping by Fehrman.

# Heuristic: columns after the 12th column are targets
if raw.shape[1] >= 30:
    predictors = raw.columns[:12].tolist()
    targets = raw.columns[12:].tolist()
else:
    # fallback: try names
    predictors = ['age','gender','education','country','ethnicity','Nscore','Escore','Oscore','Ascore','Cscore','Impulsivity','Sensation']
    targets = [c for c in raw.columns if c not in predictors]

print('\nPredictors detected (first 12):', predictors)
print('Targets detected (18):', targets)

# Prepare X (features) and Y (targets)
X = raw[predictors].copy()
Y = raw[targets].copy()

# If any predictor columns are non-numeric, convert/encode (gender etc.). We'll attempt to coerce to numeric
for col in X.columns:
    if X[col].dtype == 'object':
        # try to numeric-encode by factorizing
        X[col] = pd.factorize(X[col])[0]

# Some mirrors may already have quantized scaled feature values; standardize features for models that benefit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert Y to integer classes if they are strings
for col in Y.columns:
    if Y[col].dtype == 'object':
        Y[col] = pd.factorize(Y[col])[0]
    else:
        # ensure integer
        Y[col] = Y[col].astype(int)

# ------ Problem a: Select 2 out of 18 output features and predict each with a multiclass model
# We'll pick two targets: first two in targets list
t1, t2 = targets[0], targets[1]
print('\nSelected targets for (a):', t1, ',', t2)

# Split and train two different multiclass models with default params
X_train, X_test, y1_train, y1_test = train_test_split(X_scaled, Y[t1], test_size=0.25, random_state=42, stratify=Y[t1])
_, _, y2_train, y2_test = train_test_split(X_scaled, Y[t2], test_size=0.25, random_state=42, stratify=Y[t2])

# Model 1: RandomForestClassifier (multiclass out of the box)
rf1 = RandomForestClassifier(random_state=42)
rf1.fit(X_train, y1_train)
y1_pred = rf1.predict(X_test)
acc1 = accuracy_score(y1_test, y1_pred)
print(f'\n(a) Model 1 - RandomForest on {t1} default accuracy: {acc1:.4f}')
print(classification_report(y1_test, y1_pred))

# Model 2: LogisticRegression (multinomial)
lr2 = LogisticRegression(multi_class='multinomial', max_iter=1000)
lr2.fit(X_train, y2_train)
y2_pred = lr2.predict(X_test)
acc2 = accuracy_score(y2_test, y2_pred)
print(f'\n(a) Model 2 - LogisticRegression (multinomial) on {t2} default accuracy: {acc2:.4f}')
print(classification_report(y2_test, y2_pred))

# ------ Problem b: Manual optimization for one default parameter (at least 5 values) for one of previous models
# We'll tune RandomForest n_estimators: [10,50,100,200,500]
param_values = [10, 50, 100, 200, 500]
accs = []
for n in param_values:
    m = RandomForestClassifier(n_estimators=n, random_state=42)
    m.fit(X_train, y1_train)
    p = m.predict(X_test)
    accs.append(accuracy_score(y1_test, p))

print('\n(b) RandomForest n_estimators vs accuracy:')
for n,a in zip(param_values, accs):
    print(f'n_estimators={n}: accuracy={a:.4f}')

plt.figure(figsize=(8,5))
plt.plot(param_values, accs, marker='o')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy on test set')
plt.title(f'RandomForest accuracy vs n_estimators for target {t1}')
plt.grid(True)
plt.tight_layout()
plt.show()

# ------ Problem c: Train one multiclass classification model to predict 16 output features (exclude the 2 used above)
exclude = {t1, t2}
targets_16 = [t for t in targets if t not in exclude]
print('\n(c) Training MultiOutputClassifier with RandomForest to predict 16 targets (excluded:', t1, ',', t2, ')')

# Use a subset in case there are more than 16; but we expect exactly 18 targets
targets_16 = targets_16[:16]
Y16 = Y[targets_16]

X_tr, X_te, Ytr, Yte = train_test_split(X_scaled, Y16, test_size=0.25, random_state=42)

base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
multi_rf = MultiOutputClassifier(base_rf)
multi_rf.fit(X_tr, Ytr)
Ypred = pd.DataFrame(multi_rf.predict(X_te), columns=targets_16)

# Report accuracy (overall per-target) and per-class breakdown for each of the 7 classes
per_target_accuracy = {}
for col in targets_16:
    acc = accuracy_score(Yte[col], Ypred[col])
    per_target_accuracy[col] = acc
    print(f'Accuracy for {col}: {acc:.4f}')

# For each target, show class-wise support and accuracy
print('\nPer-target, per-class accuracy (for classes 0..6)')
for col in targets_16:
    print('\nTarget:', col)
    true = Yte[col].values
    pred = Ypred[col].values
    for cls in sorted(np.unique(true)):
        idx = (true == cls)
        if idx.sum() == 0:
            continue
        cls_acc = (pred[idx] == true[idx]).mean()
        print(f' class {cls}: accuracy {cls_acc:.3f} (n={idx.sum()})')

# ------ Problem d: Binary classification for 3 out of 18 output features
# We'll pick three targets (first three)
b_targets = targets[:3]
print('\n(d) Binary classification targets:', b_targets)

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

results = {}
for bt in b_targets:
    y = raw[bt].copy()
    # If y is object, map to original seven categories if present in mirrors; else assume numeric 0..6
    if y.dtype == 'object':
        y = pd.factorize(y)[0]
    else:
        y = y.astype(int)
    # Setup three binarizations
    setups = {
        'setup1': [0,2],  # class 0: Never Used (0) and Used in Last Decade (2) -> per problem statement first variant; but note UCI ordering may differ
        'setup2': [0,2,3],
        'setup3': [0,2,3,4]
    }
    # We'll create binary labels as: 0 if in listed groups else 1
    results[bt] = {}
    for sname, group_list in setups.items():
        bin_y = np.where(np.isin(y, group_list), 0, 1)
        Xtr, Xte, ytr, yte = train_test_split(X_scaled, bin_y, test_size=0.25, random_state=42, stratify=bin_y)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = accuracy_score(yte, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average='binary', zero_division=0)
        # try ROC AUC if both classes present
        try:
            probs = clf.predict_proba(Xte)[:,1]
            auc = roc_auc_score(yte, probs)
        except Exception:
            auc = np.nan
        results[bt][sname] = {'accuracy':acc, 'precision':prec, 'recall':rec, 'f1':f1, 'auc':auc}

# Print results
for bt, res in results.items():
    print(f'\nBinary results for target {bt}:')
    for s, metrics in res.items():
        print(f' {s}: acc={metrics["accuracy"]:.4f}, prec={metrics["precision"]:.4f}, rec={metrics["recall"]:.4f}, f1={metrics["f1"]:.4f}, auc={metrics["auc"]}')

# Note: depending on the mirror used, mapping of textual class labels to numeric codes (0..6) may differ. Verify mapping on your CSV and adjust group indices accordingly.

print('\nDone.\n')

# End of script
