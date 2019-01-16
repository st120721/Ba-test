"""
    main:

    Das Modul ist der Zugang zum Verlauf von Klassifikation.
    Es ruft die Methoden von "feature_extraction_rms","feature_extraction_wavelet" und "machine_learning".

"""
import os
import datetime as d
import pandas as pd

from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn import datasets
from machine_learning import Machine_Learning
from feature_extraction_wavelet import Feature_Extraction_Wavelet
from feature_extraction_rms import Feature_Extraction_RMS

def main(data_folder,feature_extraction_type,loops):
    """
        In der Methode gibt es drei Schleifen: Rohdaten, Klassifikatoren und Scoring.
        Die Ergebnisee werden im Ordner gespeichert.

        :param data_folder: Name des Ordner von Rohdaten (Ordner im "Rohdaten")
        :param fe_type: type von Feature Extraktion("wavelet","rms")
        :param loops: wie viele Male werden die Hyperparametersoptimierung durchgeführt.
    """
    # load daten
    root_folder="Rohdaten"
    data_path=root_folder+"\\"+data_folder
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
    files = os.listdir(data_path)
    for file in files:
        print("----------------START:----------------")
        print("Rohdaten: ",file)
        if not os.path.isdir(file):
            raw_data = data_path+"\\"+file

            output_path="Ergebnisse\\"+data_folder+"\\"+file+"\\"

            # clf:Klassifiktor
            for clf in ["KNN","SVM","RF","ADA"]:

                print("\nKlassifiktor: ",clf)
                # Pfad vom Ordner, in dem die Daten gespeichert werden
                output_path_temp1 = output_path+clf+ "_"+str(d.datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))+"\\"+clf+"_"+feature_extraction_type

                isExists = os.path.exists(output_path_temp1)
                if not isExists:
                    os.makedirs(output_path_temp1)


                # Feature Extraktion
                print("Feature Extraktion: ", feature_extraction_type)
                if feature_extraction_type == "wavelet":
                    # Hier kann man die "transformation_name"("DWT","WP") und "transformation_level" ändern
                    fe = Feature_Extraction_Wavelet(output_path=output_path_temp1, input_path=raw_data,
                                            transformation_type="DWT", transformation_level=4)
                if feature_extraction_type == "rms":
                    fe =Feature_Extraction_RMS(output_path=output_path_temp1, input_path=raw_data)
                features = fe.features.values
                labels = fe.labels.values.ravel()

                # iris = datasets.load_iris()
                # features = iris.data
                # labels = iris.target + 1

                # in Train-Set und Test_set geliedern
                skf = StratifiedShuffleSplit(n_splits=10, test_size=0.3, train_size=0.7)
                for train_index, test_index in skf.split(features, labels):
                    X_train, X_test = features[train_index], features[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]

                # für Neural Network Labels verarbeiten
                # 1 >>>> [1,0,0,0,0,...]
                # 2 >>>> [0,1,0,0,0,...]
                # 3 >>>> [0,0,1,0,0,...]
                if (clf == "DNN"):
                    y_train = y_train - 1
                    y_test = y_test - 1
                    y_train = np_utils.to_categorical(y_train, num_classes=17)
                    y_test = np_utils.to_categorical(y_train, num_classes=17)

                # scoring
                scorings_dict =Machine_Learning.make_score(3)
                for scoring_name,scoring_function in scorings_dict.items():
                    print("\nScoring: ",scoring_name)
                    output_path_temp2=output_path_temp1+"\\"+clf+"_"+str(scoring_name)
                    isExists = os.path.exists(output_path_temp2)
                    if not isExists:
                        os.makedirs(output_path_temp2)

                    scoring =[scoring_name,scoring_function]
                    Machine_Learning(algorithm=clf,scoring=scoring,loops=loops,output_path=output_path_temp2,
                                       features=X_train, labels=y_train)

if __name__ == "__main__":
    main("Test_Daten","wavelet",10)

