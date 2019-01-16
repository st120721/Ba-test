"""
    feature_extracion_rms:

    Das Modul ist aufgebaut, um die Features aus RMS-Daten zu extrahieren.
    Die folgende Funktionen sind realisiert:
        1. Daten laden und Features exportieren
        2. Features in der Liste von Klasse rechnen

"""

import os
import numpy as np
import pandas as pd
from scipy import stats


class Feature_Extraction_RMS:
    """
        Feature Extraction RMS:

        Hauptklasse von Modul "feature_extracion_rms"
        Eine Unterklasse "Calculate_Feature" ist aufgebaut,um die Features zu rechnen.

        Attributes
        ----------
            list_features: eine Liste von Features
            output_path: das Pfad vom Ordner ,in dem die Ergebnisse gespeichert werden.
            raw_data: Rohdaten
            features: features(Datentype:DataFrame)
            labels: labels(Datentype:DataFrame)
            features_and_labels: features and labels(Datentype:DataFrame)

    """

    list_features =["energy","entropy","mean","standard deviation","max","min","median","range"]
    def __init__(self,output_path,input_path):

        """
            Konstruktionsfunktion von der Klasse.

            Parameters
            ----------
                output_path: das Pfad vom Ordner ,in dem die Ergebnisse gespeichert werden.
                input_path: das Pfad der Dokumente von Rohndaten

        """

        # Ein Ordner wird hergestellt,when der Ordner existiert nicht.
        isExists = os.path.exists(output_path)
        if not isExists:
            os.makedirs(output_path)

        self.output_path =output_path
        self.input_path = input_path
        self.raw_data=self.load_data()
        self.features,self.labels,self.features_and_labels = self.features_extraction()
        self.wirte_to_csv()



    def load_data(self):
        """
            lade Daten

            Return
            ------
                raw_data: Rohdaten

        """
        path = self.input_path
        raw_data =pd.read_csv(path)
        return raw_data

    def wirte_to_csv(self):
        """
            Features und Labels in Excel speichern

        """
        output_data =self.features_and_labels
        output_name ="FE_"+"RMS_"+".csv"
        output_path=self.output_path+"\\"+output_name
        output_data.to_csv(output_path,index=False)

    def features_extraction(self):
        """
            Methode für Feature Extraktion von aller Beispiele

            Return
            ------
                features,labels,features_and_labels

        """
        features =[]
        data = self.raw_data
        labels = pd.DataFrame(data.ix[:, data.shape[1] - 1])
        labels.columns = ["label"]
        data =data.drop([str(data.shape[1]-1)],axis=1)

        for m in range(0,data.shape[0]):
            data_row= data[m:m+1].ix[:, 0:data.shape[1]]
            data_row = np.asarray(data_row).tolist()[0]
            features_single_instance =np.array(self.Calculate_Feature.feature_extraction_singe_instance(data_row))
            features.append(features_single_instance)
        features =pd.DataFrame(features)
        features_and_labels = features.join(labels)
        return features,labels,features_and_labels

    class Calculate_Feature:
        """
            Calculate Features:

            Die Klasse ist aufgebaut, um die Features von einem Beispiel zu rechnen.

        """
        @staticmethod
        def feature_energy(wert):
            """
                Energie rechnen

                Parameters
                ----------
                    wert: die Daten von einem Beispiel

                Return
                -------
                    energy

            """
            squ = np.square(wert)
            energy =np.sum(squ)
            return energy

        @staticmethod
        def feature_entropy(wert):

            squ = np.square(wert)
            ent = stats.entropy(squ)
            return ent

        @staticmethod
        def feature_mean(wert):
            mean=np.mean(wert)

            return mean

        @staticmethod
        def feature_standard_deviation(wert):
            std= np.std(wert)
            return std

        @staticmethod
        def feature_max(wert):
            max =np.max(wert)
            return max

        @staticmethod
        def feature_min(wert):
            min =np.min(wert)
            return min

        @staticmethod
        def feature_median(wert):
            median = np.median(wert)
            return median

        @staticmethod
        def feature_extraction_singe_instance(wert):
            """
                features von einem Beispiel rechnen und in einer Liste speichern

                Parameters
                ----------
                    wert: die Daten von einem Beispiel

                Return
                -------
                    features_single_instance, features von einem Beispiel

            """
            features_single_instance = []
            feature_energy =Feature_Extraction_RMS.Calculate_Feature.feature_energy(wert)
            features_single_instance.append(feature_energy)
            feature_ent =Feature_Extraction_RMS.Calculate_Feature.feature_entropy(wert)
            features_single_instance.append(feature_ent)
            feature_mean =Feature_Extraction_RMS.Calculate_Feature.feature_mean(wert)
            features_single_instance.append(feature_mean)
            feature_std = Feature_Extraction_RMS.Calculate_Feature.feature_standard_deviation(wert)
            features_single_instance.append(feature_std)
            feature_max =Feature_Extraction_RMS.Calculate_Feature.feature_max(wert)
            features_single_instance.append(feature_max)
            feature_min = Feature_Extraction_RMS.Calculate_Feature.feature_min(wert)
            features_single_instance.append(feature_min)
            feature_median = Feature_Extraction_RMS.Calculate_Feature.feature_median(wert)
            features_single_instance.append(feature_median)
            feature_range = feature_max-feature_min
            features_single_instance.append(feature_range)

            return features_single_instance

