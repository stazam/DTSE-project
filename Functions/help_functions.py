from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_recall_fscore_support


def knn_inputer(X):

    X[X.isnull()] = np.nan

    imputer = KNNImputer(n_neighbors=2)
    X_im = imputer.fit_transform(X)
    return pd.DataFrame(X_im, columns = list(X.columns))

def process_dat(X):
  
  X_1 = X.loc[:,['title','developer','publisher']]
  X_1 = pd.get_dummies(X_1)
  X_1 = pd.concat([X_1,X.loc[:,['year']]], axis = 1)

  return X_1


def plot_loss(model):

  plt.figure(1)
  plt.plot(model.history['loss'], label='loss')
  plt.plot(model.history['val_loss'], label='val_loss')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

  plt.figure(2)
  plt.plot(model.history['accuracy'], label='accuracy')
  plt.plot(model.history['val_accuracy'], label='val_accuracy')
  plt.ylim([0, 2])
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.grid(True) 


def draw_precision_recall_curve(model, sequences, labels):
  """
  RETURN: precision (x-axis) - recall (y-axis) curve in graph with the 
  value of AUC   
  """
    
  probs = model.predict(sequences)
  precision, recall, thresholds = precision_recall_curve(labels, probs)
  plt.plot(recall, precision, marker='.', label='My model')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend(loc="lower center")
  plt.title('Precision recall curve')
  auc_precision_recall = auc(recall, precision)
  print("AUC-PR:", auc_precision_recall)
  plt.show()



def ROC_curve(model, sequences, labels):
  """
  RETURN: ROC (FPR (x-axis) - TPR curve (y-axis)) curve in graph with the 
  value of AUC   
  """

  probs = model.predict(sequences)
  fpr, tpr, thresholds = roc_curve(labels, probs)
  auc_score = roc_auc_score(labels, probs)
  print("AUC score: ", auc_score)
  figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
  plt.plot(fpr, tpr, 'blue', label='My model')
  plt.axis([0, 1, 0, 1])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc="lower center")
  plt.title('ROC curve')
  plt.show()


def f1_m(model, sequences, labels):
  """
  RETURN: thrasholds and value of F1    
  """

  predictions = model.predict(sequences)
  precision, recall, thresholds = precision_recall_curve(labels, predictions)
  
  f1  = 2 * ( (precision * recall ) / (precision + recall))

  return (thresholds, f1)


def f1_graph(model, sequences, labels):
  """
  RETURN: graph of F1 values (on y - axis) and thresholds values on (x - axis)   
  """

  predictions = model.predict(sequences)
  precision, recall, thresholds = precision_recall_curve(labels, predictions)
  
  f1  = 2 * ( (precision * recall ) / (precision + recall))

  plt.plot(thresholds, f1[0: thresholds.shape[0]], 'blue', label='My model')
  plt.xlabel('Thresholds')
  plt.ylabel('F1')
  plt.legend(loc="lower center")
  plt.title('F1 curve')
  plt.show()



def diag_graphs(model, sequences, labels):

  probs = model.predict(sequences)
  precision, recall, thresholds = precision_recall_curve(labels, probs)
  fpr, tpr, thresholds = roc_curve(labels, probs)
  thresholds, f1 = f1_m(model, sequences, labels)  

  figure1, axis = plt.subplots(2, 2, squeeze=False)
  figure1.tight_layout(pad=3.0)
  
  # For Precision - Recall graph
  axis[0, 0].plot(recall, precision, label='My model')
  axis[0, 0].set_xlabel('Recall')
  axis[0, 0].set_ylabel('Precision')
  axis[0, 0].set_title('Precision recall curve')
  
  # For FPR - TPR graphs
  axis[0, 1].plot(fpr, tpr,  label='My model')
  axis[0, 1].set_xlabel('False Positive Rate')
  axis[0, 1].set_ylabel('True Positive Rate')
  axis[0, 1].set_title('ROC curve')
  
  # For F1 graph
  axis[1, 0].plot(thresholds, f1[0:thresholds.shape[0]],  label='My model')
  axis[1, 0].set_xlabel('Thresholds')
  axis[1, 0].set_ylabel('F1')
  axis[1, 0].set_title('F1 curve')

  # Combine all the operations and display
  plt.show()

  auc_score = roc_auc_score(labels, probs)
  print("AUC score for (TPR, FPR) - graph: ", round(auc_score,4))

  auc_precision_recall = auc(recall, precision)
  print("AUC-PR for (precision - recall) graph:", round(auc_precision_recall,4))


def XGB_function(X_train, y_train, X_test, y_test, X):
  # 0 - home, 1 - draw, 2 - away

  xgb_cl = xgb.XGBClassifier(n_estimators=15,learning_rate=0.5,max_delta_step=5)
  xgb_cl.fit(X_train,y_train)

  if ( not(X_test is None) and not(y_test is None)):
    print('Accuracy of Extreme boosted classifier on training set: {:.2f}'.format(xgb_cl.score(X_train, y_train)))
    print('Accuracy of Extreme boosted classifier on test set: {:.2f}'.format(xgb_cl.score(X_test, y_test)))

  return xgb_cl.predict(X), xgb_cl.predict_proba(X)


def LogisticRegression_function(X_train, y_train, X_test, y_test, X):

  logreg = LogisticRegression()
  logreg.fit(X_train, y_train)

  if ( not(X_test is None) and not(y_test is None)):
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
    print('Accuracy of Logistic regression on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

  return logreg.predict(X), logreg.predict_proba(X)

def DecisionTree_function(X_train, y_train, X_test, y_test, X):
  # 0 - home, 1 - draw, 2 - away

  clf = DecisionTreeClassifier()
  clf.fit(X_train, y_train)

  if ( not(X_test is None) and not(y_test is None)):
    print('Accuracy of DecisionTree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of DecisionTree on test set: {:.2f}'.format(clf.score(X_test, y_test)))

  return clf.predict(X), clf.predict_proba(X)





