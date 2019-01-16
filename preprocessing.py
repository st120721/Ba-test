"""
    preprocessing:

    Das Modul ist aufgebaut, vor Machine Learning die Rohdaten zu verarbeiten.
    Es gibt zwei Klassen: preprocessing_arena_daten und preprocessing_simulation_daten
    Die folgende Funktionen sind realisiert:
        1. Daten laden und schneiden
        2. SNR rechnen
        3. Signal mit AWGN(additive Gaussian noise) erstellen

"""
import pandas as pd
import numpy as np
from scipy import stats



class preprocessing_arena_daten:

    # def load_data(self,data_path,chunksize):
    # def load_data(self):
    #     data_path="Rohdaten\\ARENA_Daten\\20180720_E_TR1_1_DftL1C01.csv"
    #     data_path = "Rohdaten\\Test_Daten\\Testdata_1700.csv"
    #
    #     chunksize=200
    #     chunks =pd.read_csv(data_path,sep=",",header=[0],chunksize=chunksize)
    #     # for chunk in chunks:
    #     #     pass
    #     return chunks.get_chunk(1)



    # chunk.next()
    @staticmethod
    def calculate_snr(sigan,noise):

        return



class preprocessing_simulation_daten:

    @staticmethod
    def signal_with_awgn(signal, snr):
        """
            Signal mit AWGN(additive Gaussian noise) erstellen

            Parameters
            ----------
                singal: Orginalsignal
                snr: SNR(dB)

            Return
            ------
                signal_with_noise: Signal mit AWGN

        """
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        noise = np.random.randn(len(signal)) * np.sqrt(npower)
        signal_with_noise = signal + noise
        return signal_with_noise

    def waveform_to_rms(self,data):
        """
            Wellenform-Daten in RMS-Daten umsetzen

            Parameters
            ----------
                data: Name von Wellenfor

            Return
            ------
                signal_with_noise: Signal mit AWGN

        """
        segmente_num = 10
        segmente_lange=50
        rms_list = []
        for i in np.arange(1,11,1,dtype=int):
            data_temp = data[(i-1)*5:i*5]
            rms =((np.square(data_temp)).mean()) ** 0.5
            rms_list.append(rms)

        return rms_list

test=()
data=test.load_label()
