import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Configurable parameters for
NUMCAT = 10          # number of categories to examine
DELTAS = True        # wether to include deltas or not


TEST_SIZE = 0.25     # proportion of the dataset to include in the test split.

#C = 1                # Penalty parameter C of the error term.
#GAMMA = 1/N_FEATURES # Kernel coefficient

# Load data
DATA = np.load("../data/data_"+str(NUMCAT)+"_categories_deltas_"+str(DELTAS)
               +".npy")
CATEGORY = DATA[:, 0]
FEATURES = DATA[:, 1:]
#N_FEATURES = FEATURES.shape[1]

# Scale features
SCALER = StandardScaler()
X_SCALED = SCALER.fit_transform(FEATURES)

# Shuffle and split into train and test sets
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X_SCALED, CATEGORY,
                                                    test_size=TEST_SIZE)

rf = RandomForestClassifier(bootstrap=True,
            max_depth=300, max_features='log2',
            min_samples_split=12,
            n_estimators=1000, n_jobs=-1,
            oob_score=True)


rf.fit(X_TRAIN, Y_TRAIN)
Y_PRED = rf.predict(X_TEST)
error_rate = np.average(Y_TEST != Y_PRED)
print("Random Forests - Test set accuracy rate: ", 1 - error_rate)

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
          'pop', 'reggae', 'rock']
CLASSIFICATION_REPORT = classification_report(Y_TEST, Y_PRED,
                                              target_names=GENRES)

print("Classification Report:\n", CLASSIFICATION_REPORT)

#Confusion matrix
CONFUSION_MATRIX = confusion_matrix(Y_TEST, Y_PRED)
print(CONFUSION_MATRIX)
