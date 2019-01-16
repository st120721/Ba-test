"""
    machine_learning:

    Das Modul ist aufgebaut,um beste Feature-Set und beste Parameter zu suchen.
    Die folgende Funktionen sind realisiert:
        1. Ergebnisse exportieren
        2. Feature Selektion
        3. Wertbereich von Parameters definieren und Hyperparameters optimieren
        4. eine Dictionary von Scorings erstellen

    Toolbox: mlxtend für Feature Selector
             sklearn, keras, tensorflow für Machine Learning Algorithmen
             hyperopt für Hyperparameter

"""
import pandas as pd
import numpy as np
import os
import datetime as d
import csv
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import make_scorer, average_precision_score,f1_score,recall_score,precision_score,roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier


class Machine_Learning:
    """
          Machine_Learning:

          Hauptklasee vom Modul "Machine Learning"
          Und eine Unterklasse "Neural Network" ist aufgebaut,um ein Neutral Network(estimator) zu erstellen.

          Attributes
          ----------
              type_sequential_feature_selector: eine Liste von Feature Selector
              sfs_param_space: Dictionary von Parameter von Sequential Feature Selector
              list_ml_algorithms: eine Liste von Machine Learning Algorithmen
              knn_dict: Dictionary von KNN(parameter und estimator)
              svm_dict: Dictionary von SVM(parameter und estimator)
              rf_dict: Dictionary von Random Forest(parameter und estimator)
              adaboost_dict: Dictionary von Adaboost(parameter und estimator)
              nn_dict: Dictionary von Neural Network(parameter und estimator)
              algorithms_dict: Dictionary von aller Algorithmen

    """
    type_sequential_feature_selector = ["Sequential Forward Selection (SFS)", "Sequential Backward Selection (SBS)",
                                        "Sequential Forward Floating Selection(SFFS)",
                                        "Sequential Backward Floating Selection(SBFS)"]
    sfs_param_space = {"k_features": hp.choice("k_features", np.arange(1, 10, 1, dtype=int).tolist()),
                       "forward": hp.choice("forward", [True,False]),
                       "floating": hp.choice("floating", [True, False]),
                       }

    list_ml_algorithms = ["Support Vector Machines(SVM)", "K Nearest Neighbors(KNN)",
                          "Random Forests(RF)", "Bayes(BAYES)", "Artificial Neural Network(NN)",

                          ]


    knn_dict = dict(
        name="KNN",
        parameters_list=["n_neighbors"],
        parameters_grid={'n_neighbors': hp.choice('n_neighbors', np.arange(1, 10, 1, dtype=int).tolist())},
        estimator=KNeighborsClassifier(),
    )
    svm_dict=dict(
    name="SVM",
    parameters_list=["kernel","C","gamma"],
    parameters_grid={'kernel': hp.choice('kernel', ['rbf']),
                                         # ['linear', 'sigmoid', 'poly', 'rbf']),
                     'C': hp.uniform('C', 0, 10),
                    'gamma': hp.uniform('gamma', 0.0001, 5),},
    estimator = SVC(),

    )
    rf_dict=dict(
        name ="RF",
        parameters_list=['n_estimators','max_depth', "min_samples_split",'min_samples_leaf','criterion'],
        parameters_grid={
                          'n_estimators': hp.choice('n_estimators', range(1, 100)),
                          'max_depth': hp.choice('max_depth', range(10, 100)),
                          # 'min_samples_split': hp.choice('min_samples_leaf', np.arange(2, 10, 2, dtype=int).tolist()),
                          # 'min_samples_leaf':  hp.choice('min_samples_leaf', np.arange(1, 5, 1, dtype=int).tolist()),
                           'criterion': hp.choice('criterion', ["gini", "entropy"])},
        estimator = RandomForestClassifier()
    )
    adaboost_dict=dict(
        name="ADA",
        parameters_list=[ 'n_estimators','learning_rate'],
        parameters_grid={ 'n_estimators': hp.choice('n_estimators', range(1, 100)),

                          'learning_rate': hp.uniform('learning_rate', 0.0001, 1)

                         },
        estimator=AdaBoostClassifier()
    )

    nn_dict=dict(

    )
    algorithms_dict =dict(
        KNN=knn_dict,
        SVM =svm_dict,
        RF=rf_dict,
        ADA=adaboost_dict,
    )





    def __init__(self,algorithm,scoring,loops, features, labels,output_path):
        """
            Konstruktionsfunktion von der Klasse

            Parameters
            ---------
                algorithm: der gewählten Algorithmus
                scoring: Scoring für Hyperparameter
                features: Features nach Features Extraktion
                labels: Labels nach Features Extraktion
                output_path: das Pfad vom Ordner ,in dem die Ergebnisse gespeichert werden.

        """
        self.algorithm=algorithm
        self.features = features
        self.labels = labels
        self.loops = loops
        self.scoring_name =scoring[0]
        self.scoring_function = scoring[1]
        self.output_path =output_path
        self.result = self.tuning_parameter()
        self.wirte_result_to_txt()
        self.wirte_selected_feature_to_csv()

    def wirte_result_to_txt(self):
        """
            die besten Hyperparameter für das gewälten Scoring in Txt speichern

        """
        output_path = self.output_path
        isExists = os.path.exists(output_path)
        if not isExists:
            os.makedirs(output_path)
        file_name = self.algorithm + "_" + str(self.scoring_name) + ".txt"
        output_path = self.output_path + "\\" + file_name
        with open(output_path, 'w' ) as f:
            for key, value in self.result.items():
                info = key + ": " + str(value)+"\n"
                f.write(info)

    
    def wirte_selected_feature_to_csv(self):
        """
            gewälte Features nach Feauture Selektion in Excel speichern

        """
        output_path = self.output_path
        isExists = os.path.exists(output_path)
        if not isExists:
            os.makedirs(output_path)
        file_name = "selected_features_" + str(self.scoring_name) + ".csv"
        output_path = output_path + "\\" + file_name
        features = self.features[:, list(self.result["selected_features_idx"])]
        features=pd.DataFrame(features,columns=list(self.result["selected_features_idx"]))
        features.to_csv(output_path, index=False)

    def tuning_parameter(self):
        """
            Die Methode ist nach einem Beispiel vom Toolbox"hyperopt"aufgebaut, um Hyperparameter zu optimieren.
            Sie wird nach dem Anzahl von "loop"  die Methode "tuning_parameter_one_loop" rufen
            Die Daten im Verlauf von der Hyperparametersuche werden in Excel gespeichert.

            Return
            ------
                result: Dictionary von Ergebnisse, das Hyperparameter, Dauer, Score usw. enthält

        """
        algorithm_dict_temp = self.algorithms_dict[self.algorithm]
        scoring_name =self.scoring_name
        scoring_function = self.scoring_function

        # die Bezeichnung von der Reihen in Excel hinzufügen
        output_path = self.output_path
        isExists = os.path.exists(output_path)
        if not isExists:
            os.makedirs(output_path)
        file_name = self.algorithm + "_" + str(self.scoring_name) + "_" + "tunning_parameter_info.csv"
        writer_path = output_path + "\\" + file_name
        with open(writer_path, "x" and "w")as f:
            writer = csv.writer(f)
            loop_info_list = ["loop", str(self.scoring_name), "SFS", "num selected features",
                              "feature idx"]+(algorithm_dict_temp["parameters_list"])

            writer.writerow(loop_info_list)

        # pre-processing:scale
        features = preprocessing.scale(self.features)
        labels =self.labels

        # parameter space
        sfs_param_space = self.sfs_param_space
        algorithm_param_space = algorithm_dict_temp["parameters_grid"]
        parma_space = dict(sfs_param_space, **algorithm_param_space)

        # loop

        temp_loop = 0
        # Anfangswerte von Ergebnisse
        result = {
            "time start": d.datetime.now(),
            "time end": None,
            "duration": None,
            "loops": 0,
            "scoring": scoring_name,
            "best_score": 0,
            "sfs_type": None,
            "num_features": 0,
            "selected_features_idx": None,
        }
        result = dict(result, **(dict.fromkeys(algorithm_dict_temp["parameters_list"], None)))


        def tuning_parameter_one_loop(params):
            """
                eine Schleife von Hyperparameteroptimierung

                Parameters
                ----------
                    params: parameter space

                Return
                ------
                    Dictionary von "loss" und "status", die von Hyperopt definiert.

            """
            nonlocal temp_loop
            temp_loop = temp_loop + 1
            print("loop: ",temp_loop)

            clf = algorithm_dict_temp["estimator"]

            # parameter von Klassifikator  stellen
            for keys1, values1 in params.items():

                for keys2, values2 in clf.__dict__.items():
                    if keys1 == keys2:
                        clf.__dict__[keys1] = values1


            # Sequential Feature Selector
            sfs = SequentialFeatureSelector(estimator=clf, scoring=scoring_function,
                                              n_jobs=-1,
                                              cv=5, )
            for keys1, values1 in params.items():
                for keys2, values2 in sfs.__dict__.items():
                    if keys1 == keys2:
                        sfs.__dict__[keys1] = values1

            forward=sfs.forward
            floating =sfs.floating
            if forward == True and floating == False:
                sfs_type = "SFS"
            elif forward == False and floating == False:
                sfs_type = "SBS"
            elif forward == True and floating == True:
                sfs_type = "FSFS"
            elif forward == False and floating == True:
                sfs_type = "FSBS"


            sfs.fit(features, labels)
            score = sfs.k_score_

            # die Ergebnisse von einem Kreis in der Liste "loop_info" hinzufügen
            loop_info = [temp_loop,score, sfs_type, sfs.k_features, sfs.k_feature_idx_]
            for param in algorithm_dict_temp["parameters_list"]:
                    loop_info.append(clf.__getattribute__(param))

            #  die Information von "loop info"in Excel speichern
            with open(writer_path, "a")as f:
                writer = csv.writer(f)
                writer.writerow(loop_info)

            # Informationen werden in "result" gespeichert, nur wenn sie besser als die Daten in "result" sind.
            if score > result["best_score"]:
                result["best_score"] = score
                result["sfs_type"] = sfs_type
                result["num_features"] = sfs.k_features
                result["selected_features_idx"] = sfs.k_feature_idx_
                for keys1, values1 in params.items():
                    for key2, values2 in result.items():
                        if keys1 == key2:
                            result[keys1] = values1

            return {'loss': -score, 'status': STATUS_OK}

        # eine Methode von der Hyperparameteroptimierung im Toolbox "hyperopt"
        trials =Trials()

        print("Hyperparameter optimieren")
        fmin(fn=tuning_parameter_one_loop, space=parma_space, algo=tpe.suggest,
             max_evals=self.loops,
             trials=trials)


        result["time end"] = d.datetime.now()
        result["duration"] = result["time end"] - result["time start"]
        result["loops"] = temp_loop

        return result

    @staticmethod
    def make_score(klasse_num):
        """
            Liste von Scroing erstellen
            Zur zeit gibt es zwei Teile.
            Ein Teil ist für die zusammenfassende Scoring vom Modell,z.B. accuracy,f1 score(macro),f1 score(micro) usw.
            Anderer Teil ist für jeder Klasse,z.B.recall_1,recall_2 usw.

            Parameter
            ---------
                klasse_num: Anzahl von Klasse

            Return
            ------
                scorings_dict: Dictionary von Scroing

        """
        scorings_dict = {"accuracy": "accuracy",
                       # "f1_score_macro": make_scorer(f1_score, average="macro"),
                       # "f1_score_micro": make_scorer(f1_score, average="micro"),
                       "recall_macro": make_scorer(recall_score, average="macro"),
                       "recall_micro": make_scorer(recall_score, average="micro"),
                       "precision_macro": make_scorer(precision_score, average="macro"),
                       "precision_micro": make_scorer(precision_score, average="micro"),
                       }
        for i in np.arange(1,klasse_num+1,1,dtype=int):
                temp_dict = {"recall_" + str(i): make_scorer(recall_score, labels=[i], average=None),
                             "precision_" + str(i): make_scorer(precision_score, labels=[i], average=None),
                              # "f1_score_"+ str(i): make_scorer(f1_score, labels=[i], average=None)
                             }
                scorings_dict = dict(scorings_dict, **(temp_dict))
        # scorings_name_list =[]
        # scorings_function_list=[]
        # for keys,values in scorings_dict.items():
        #     scorings_name_list=scorings_name_list.append(keys)
        #     scorings_function_list= scorings_function_list.append(values)
        return scorings_dict

    # class Deep_Neural_Network:
    #
    #     def creat_estimator(self):
    #         # grid search epochs, batch size and optimizer
    #
    #         optimizers = ['rmsprop', 'adam']
    #         init = ['glorot_uniform', 'normal', 'uniform']
    #         epochs = [50, 100, 150]
    #         batches = [5, 10, 20]
    #         param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
    #
    #
    #         model = Sequential()
    #
    #         model.add(Dense(64, activation='relu', input_dim=20))
    #         model.add(Dropout(0.5))
    #         model.add(Dense(64, activation='relu'))
    #         model.add(Dropout(0.5))
    #         model.add(Dense(10, activation='softmax'))
    #
    #         sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #         model.compile(loss='categorical_crossentropy',
    #                       optimizer=sgd,
    #                       metrics=['accuracy'])
    #
    #         model.fit(x_train, y_train,
    #                   epochs=20,
    #                   batch_size=128)
    #         score = model.evaluate(x_test, y_test, batch_size=128)
    #
    #
    #
    #         clf =KerasClassifier(model)
    #         return clf
    
       




# iris = datasets.load_iris()
# features=pd.DataFrame(iris.data)
# labels = iris.target
# # scoring =Scoring()
# # print(scoring.scoring)
# # for keys,values in scoring.scoring.items():
#
# test =ML_Algorithms(algorithm="KNN",loops=10,features=features,
#                     labels=labels,output_path="test")
# test.main()

