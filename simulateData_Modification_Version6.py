# Erzeugt unterschiedliche Störfälle und speichert sie im angegebenen Pfad als numpy-Array
# (als Vorlage diente die MATLAB-Datei 'simulateData.m'

import math
import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import time
import pickle
import json
import csv

# ---------------------------------------------------------------------------------------------
#                   HIER BITTE DEN PFAD ZUM ABSPEICHERN ANGEBEN
# ---------------------------------------------------------------------------------------------
# Save as Array mit Datum und Uhrzeit im Dateiname --> Laden der Datei mit 'np.load(filename)'
path = 'test'
# Variable True: alle Informationen der Erstellung werden in der Konsole
# ausgegeben; False: keine Informationen werden ausgegeben
outputVariables = True
# samplingFrequency fuer die gewuenschte Abtastrate einstellen.
samplingFrequency = 50000
# Einstellen wie viele Perioden pro Fehlerfall in der Messung vorhanden sein sollen
cycles = 5
# Frequenz des Netzes
f = 50
# Einstellen wie viele Fehlerfaelle pro Fehlerklasse generiert werden sollen
casesPerClass = 200
# Anzahl der harmonischen Graden
grade = 9
#pure Sinwave
x = np.linspace(1 / samplingFrequency, cycles / f, samplingFrequency / f * cycles)
sinwave = np.sin(2 * np.pi * f * x)
# gewünchste Kombination
kombination_list = [ 'Überspannung','Harmonische'] # Name der Ströung angeben
# ---------------------------------------------------------------------------------------------


def mysinwave(A, f, sf, cycles, alpha, t1, t2):
    # Creates a sin wave. You can add deviation with the constant alpha.
    if (-0.1 <= alpha) and (alpha <= 0.1):
        # normal
        signalclass = 1
    elif (-0.9 < alpha) and (alpha < -0.1):
        # sag
        signalclass = 2
    elif (0.1 < alpha) and (alpha < 0.8):
        # swell
        signalclass = 3
    elif (-1 <= alpha) and (alpha <= -0.9):
        # Interruption
        signalclass = 4
    x = np.linspace(1 / sf, cycles / f, sf / f * cycles)
    sinwave = np.sin(2 * np.pi * f * x) * A * (
            1 + alpha * (np.heaviside((x - t1), 0.50) - np.heaviside((x - t2), 0.50)))
    sinwave = np.append(sinwave, signalclass)

    return sinwave

def myharmonic(A, grade, f, sf, cycles):

    summe_of_coefficient = 0
    x = np.linspace(1 / sf, cycles / f, sf / f * cycles)
    harmonic = 0


    for i in range(2, grade + 1):
        alpha = np.random.random() * 0.1 + 0.05
        summe_of_coefficient += np.power(alpha, 2)
        harmonic += alpha * np.sin(i * w * x)

    alpha1 = math.sqrt(1 - summe_of_coefficient)
    harmonic = alpha1 * np.sin(w * x) + harmonic


    harmonic = np.append(A * harmonic,5)

    return harmonic

def mysinwaveWithHarmonic(A, f, sf, cycles, alpha, t1, t2, grade):
    #Creates a sin wave. You can add deviation with the constant alpha.
    if (-0.9 < alpha) and (alpha < -0.1):
        # sag
        signalclass = 6
    elif (0.1 < alpha) and (alpha < 0.8):
        # swell
        signalclass = 7
    elif (-1 <= alpha) and (alpha <= -0.9):
        # Interruption
        signalclass = 8

    summe_of_coefficient = 0
    x = np.linspace(1 / sf, cycles / f, sf / f * cycles)
    harmonic = 0

    for i in range(2, grade + 1):
        alpha_harmonisch = np.random.random() * 0.1 + 0.05
        summe_of_coefficient = np.power(alpha_harmonisch, 2) + summe_of_coefficient
        harmonic = alpha_harmonisch * np.sin(i * w * x) + harmonic

    alpha1 = math.sqrt(1 - summe_of_coefficient)
    harmonic = alpha1 * np.sin(w * x) + harmonic

    sinwaveWithHarmonic = A * (1 + alpha * (np.heaviside((x - t1), 0.50) - np.heaviside((x - t2), 0.50))) * harmonic
    sinwaveWithHarmonic = np.append(sinwaveWithHarmonic,signalclass)
    return sinwaveWithHarmonic



def myflicker(A, f, sf, cycles, alpha, t1, t2, alphaf, beta):
    if (-0.1 <= alpha) and (alpha <= 0.1):
        # flicker
        signalclass = 9
    elif (-0.9 < alpha) and (alpha < -0.1):
        # flicker with sag
        signalclass = 15
    elif (0.1 < alpha) and (alpha < 0.8):
        # flicker with swell
        signalclass = 16
    elif (-1 <= alpha) and (alpha <= -0.9):
        # flicker with Interruption
        signalclass = 17
    x = np.linspace(1 / sf, cycles / f, sf / f * cycles)
    flicker = np.sin(2 * np.pi * f * x) * A * (1 + alphaf * np.sin(2 * np.pi * f * beta * x)) * \
              (1 + alpha * (np.heaviside((x - t1), 0.50) - np.heaviside((x - t2), 0.50)))
    flicker = np.append(flicker, signalclass)
    return flicker


def myOscillatorytransient(A, alpha, fn, t1, t2, tau, f, sf, cycles, delta):
    wn = 2 * np.pi * fn
    xdelta = t1 - np.log(delta / alpha) * tau;
    if xdelta < t2:
        myend = xdelta
    elif t2 < xdelta:
        myend = t2
    x = np.linspace(1 / sf, cycles / f, sf / f * cycles)
    oscillatoryTransient = A * (np.sin(2 * np.pi * f * x) + alpha * np.exp(-((x - t1) / tau)) * np.sin(wn * x) *
                                (np.heaviside((x - t1), 0.50) - np.heaviside((x - t2), 0.50)))
    oscillatoryTransient = np.append(oscillatoryTransient, 10)
    return oscillatoryTransient


def myimpulsiveTransient(A, alpha, t1, t3, risetime, duration, f, sf, cycles, delta):
    x = np.linspace(1 / sf, cycles / f, sf / f * cycles)
    t2 = risetime + t1

    tau = -(duration) / math.log(1 / 2);
    xdelta = t1 - math.log(delta / alpha) * tau + risetime;
    if xdelta < t3:
        myend = xdelta
    elif t3 < xdelta:
        myend = t3

    impulsiveTransient = A * (
            np.sin(2 * np.pi * f * x) + (alpha * ((np.heaviside((x - t1), 0.50) * (x - t1) / (t2 - t1))
                                                  - (np.heaviside((x - t2), 0.50) * ((x - t2) / (t2 - t1) + 1)) +
                                                  ((np.heaviside((x - t2), 0.50) * np.exp(-((x - t2) / tau)))
                                                   - (np.heaviside((x - t3), 0.50) * np.exp(
                                                              -((x - t2) / tau)))))));
    impulsiveTransient = np.append(impulsiveTransient, 11)
    return impulsiveTransient


def myNotch(A, f, sf, cycles, K, t1, t2, pos):
    # Creates a sin wave. You can add deviation with the constant alpha, either
    # the whole time or in the intervall from t1 to t2
    if pos == -1:
        # notch
        signalclass = 12
    elif pos == 1:
        # spike
        signalclass = 13

    x = np.linspace(1 / sf, cycles / f, sf / f * cycles)

    notch = A * (np.sin(2 * np.pi * f * x) + pos * np.sign(np.sin(2 * np.pi * f * x)) *
                 (K * (np.heaviside((x - (t1 + 0.02 * 0)), 0.50) - np.heaviside((x - (t2 + 0.02 * 0)), 0.50)) +
                  K * (np.heaviside((x - (t1 + 0.02 * 1)), 0.50) - np.heaviside((x - (t2 + 0.02 * 1)), 0.50)) +
                  K * (np.heaviside((x - (t1 + 0.02 * 2)), 0.50) - np.heaviside((x - (t2 + 0.02 * 2)), 0.50)) +
                  K * (np.heaviside((x - (t1 + 0.02 * 3)), 0.50) - np.heaviside((x - (t2 + 0.02 * 3)), 0.50)) +
                  K * (np.heaviside((x - (t1 + 0.02 * 4)), 0.50) - np.heaviside((x - (t2 + 0.02 * 4)), 0.50)) +
                  K * (np.heaviside((x - (t1 + 0.02 * 5)), 0.50) - np.heaviside((x - (t2 + 0.02 * 5)), 0.50)) +
                  K * (np.heaviside((x - (t1 + 0.02 * 6)), 0.50) - np.heaviside((x - (t2 + 0.02 * 6)), 0.50)) +
                  K * (np.heaviside((x - (t1 + 0.02 * 7)), 0.50) - np.heaviside((x - (t2 + 0.02 * 7)), 0.50)) +
                  K * (np.heaviside((x - (t1 + 0.02 * 8)), 0.50) - np.heaviside((x - (t2 + 0.02 * 8)), 0.50)) +
                  K * (np.heaviside((x - (t1 + 0.02 * 9)), 0.50) - np.heaviside((x - (t2 + 0.02 * 9)), 0.50))))
    notch = np.append(notch, signalclass)
    return notch

def myflickerWithHarmonic(A, grade, f, sf, cycles, alphaf, beta):

    summe_of_coefficient = 0
    x = np.linspace(1 / sf, cycles / f, sf / f * cycles)
    harmonic = 0

    for i in range(3, grade + 2, 2):
        alpha = np.random.random() * 0.1 + 0.05
        summe_of_coefficient = np.power(alpha, 2) + summe_of_coefficient
        harmonic = alpha * np.sin(i * w * x) + harmonic

    alpha1 = math.sqrt(1 - summe_of_coefficient)
    harmonic = alpha1 * np.sin(w * x) + harmonic

    flickerWithHarmonic = A * harmonic * (1 + alphaf * np.sin(2 * np.pi * f * beta * x))
    flickerWithHarmonic = np.append(flickerWithHarmonic, 14)

    return flickerWithHarmonic


def creatKombination(A,delta:list,kombinaton:list,Dictionary:dict):

    List = [Dictionary[name] for name in kombinaton]

    new_Wave_list = []

    for i in range(len(List)):
        if i == 0 :
            a = (List[i] - 2) * casesPerClass
            #print(a)
            b = (List[i] - 1) * casesPerClass
            #print(b)
            # print('hallo i am here')
            for j in range(a,b):
                new_Wave_list.append(delta[j])
                # print('hallo i am here')
        else:
            a = (List[i] - 2) * casesPerClass
            b = (List[i] - 1) * casesPerClass
            l = 0
            for m in range(a,b):
                new_Wave_list[l] = new_Wave_list[l] + delta[m]
                l = l + 1

    for n in range(len(new_Wave_list)):
        new_Wave_list[n] = A * (new_Wave_list[n] + sinwave)
        new_Wave_list[n] = np.append(new_Wave_list[n],18)

    return new_Wave_list



def plotData(f, sf, cycles, A, casesPerClass,delta,kombination,dictionary):
    # Erzeugung einer x-Achse in Sekunden
    x = np.linspace(1 / sf, cycles / f, sf / f * cycles)

    a = creatKombination(A,delta,kombination,dictionary)

    # Figur erstellen mit allen unterschiedlichen Signaltypen (alle mit 'visible=False' zu Beginn ausgeblendet,
    # 'lw' zeigt die Linienstärke an)
    fig, ax = plt.subplots()
    l0, = ax.plot(x, feature[0 * casesPerClass][0:len(x)], visible=False, lw=1)
    l1, = ax.plot(x, feature[1 * casesPerClass][0:len(x)], visible=False, lw=1)
    l2, = ax.plot(x, feature[2 * casesPerClass][0:len(x)], visible=False, lw=1)
    l3, = ax.plot(x, feature[3 * casesPerClass][0:len(x)], visible=False, lw=1)
    l4, = ax.plot(x, feature[4 * casesPerClass][0:len(x)], visible=False, lw=1)
    l5, = ax.plot(x, feature[5 * casesPerClass][0:len(x)], visible=False, lw=1)
    l6, = ax.plot(x, feature[6 * casesPerClass][0:len(x)], visible=False, lw=1)
    l7, = ax.plot(x, feature[7 * casesPerClass][0:len(x)], visible=False, lw=1)
    l8, = ax.plot(x, feature[8 * casesPerClass][0:len(x)], visible=False, lw=1)
    l9, = ax.plot(x, feature[13 * casesPerClass][0:len(x)], visible=False, lw=1)
    l10, = ax.plot(x, feature[14 * casesPerClass][0:len(x)], visible=False, lw=1)
    l11, = ax.plot(x, feature[15 * casesPerClass][0:len(x)], visible=False, lw=1)
    l12, = ax.plot(x, feature[16 * casesPerClass][0:len(x)], visible=False, lw=1)
    l13, = ax.plot(x, feature[9 * casesPerClass][0:len(x)], visible=False, lw=1)
    l14, = ax.plot(x, feature[10 * casesPerClass][0:len(x)], visible=False, lw=1)
    l15, = ax.plot(x, feature[11 * casesPerClass][0:len(x)], visible=False, lw=1)
    l16, = ax.plot(x, feature[12 * casesPerClass][0:len(x)], visible=False, lw=1)
    l17, = ax.plot(x,a[0][0:len(x)], visible=False, lw=1)

    s = ''

    for k in kombination_list:
        s = s + k

    # Abstand des Plots vom linken Rand des Fensters (vergleiche mit CheckBox)
    plt.subplots_adjust(left=0.4)

    # alle Titel in der Legende zur Auswahl
    beschriftung = ['Sinus mit Intensität', 'Spannungseinbruch', 'Überspannung', 'Unterbrechung',
                    'Harmonische', 'Harmonische + Spannungseinbruch', 'Harmonische + Überspannung',
                    'Harmonische + Unterbrechung', 'Flicker', 'Flicker mit Harmonischen', 'Flicker + Spannungseinbruch',
                    'Flicker + Überspannung', 'Flicker + Unterbrechung', 'Oszillator Transienten',
                    'Impulsive Transiente', 'Periodischer Notch', 'Periodischer Spike',s]

    # Position der Checkbox im Fenster, alle Angaben in % zu Fenster [Rand_links[%] Rand_unten[%] Breite[%] Höhe[%]]
    rax = plt.axes([0.02, 0.02, 0.3, 0.96])
    check = CheckButtons(rax, (beschriftung[0], beschriftung[1], beschriftung[2], beschriftung[3], beschriftung[4],
                               beschriftung[5], beschriftung[6], beschriftung[7], beschriftung[8], beschriftung[9],
                               beschriftung[10], beschriftung[11], beschriftung[12], beschriftung[13],
                               beschriftung[14], beschriftung[15], beschriftung[16],beschriftung[17]),
                         (False, False, False, False, False, False, False, False, False, False, False,
                          False, False, False, False, False, False,False))


    # Funktion, die die Sichtbarkeit des jeweiligen Signals toggelt (TRUE -> FALSE -> TRUE -> ...)
    def func(label):
        if label == beschriftung[0]:
            l0.set_visible(not l0.get_visible())
        elif label == beschriftung[1]:
            l1.set_visible(not l1.get_visible())
        elif label == beschriftung[2]:
            l2.set_visible(not l2.get_visible())
        elif label == beschriftung[3]:
            l3.set_visible(not l3.get_visible())
        elif label == beschriftung[4]:
            l4.set_visible(not l4.get_visible())
        elif label == beschriftung[5]:
            l5.set_visible(not l5.get_visible())
        elif label == beschriftung[6]:
            l6.set_visible(not l6.get_visible())
        elif label == beschriftung[7]:
            l7.set_visible(not l7.get_visible())
        elif label == beschriftung[8]:
            l8.set_visible(not l8.get_visible())
        elif label == beschriftung[9]:
            l9.set_visible(not l9.get_visible())
        elif label == beschriftung[10]:
            l10.set_visible(not l10.get_visible())
        elif label == beschriftung[11]:
            l11.set_visible(not l11.get_visible())
        elif label == beschriftung[12]:
            l12.set_visible(not l12.get_visible())
        elif label == beschriftung[13]:
            l13.set_visible(not l13.get_visible())
        elif label == beschriftung[14]:
            l14.set_visible(not l14.get_visible())
        elif label == beschriftung[15]:
            l15.set_visible(not l15.get_visible())
        elif label == beschriftung[16]:
            l16.set_visible(not l16.get_visible())
        elif label == beschriftung[17]:
            l17.set_visible(not l17.get_visible())
        plt.draw()

    # Aufrufen der Funktion 'func' bei Klicken
    check.on_clicked(func)

    # Amplitude des normalen Sinus mit dünner Linie ('lw=1') einzeichnen
    ax.plot([1 / sf, 1 / f * cycles], [A, A], lw=1)
    ax.plot([1 / sf, 1 / f * cycles], [-A, -A], lw=1)

    # Achsenbeschriftung
    ax.set_xlabel('Zeit in [s]')
    ax.set_ylabel('Spannun in [V]')
    # Gitternetzlinien anzeigen
    ax.grid(True)

    # Anzeige des Plots (!!! ohne diese Funktion wird der Plot nicht angezeigt !!!)
    plt.show()


if __name__ == '__main__':
    dateTimeUnix = int(time.time() * 1000000000)

    w = 2 * np.pi * f
    # Magnitude des Netzes
    A =  1

    #Default A = 230*math.sqrt(2)

    # Wörterbuch mit allen Informationen erstellen zur Abspeicherung
    # Dateigröße wird unverhältnismäßig groß, da numpy-Array, nur floats will
    informationDict = {
        "frequency": f,
        "cycles": cycles,
        "samplingFrequency": samplingFrequency,
        "amplitude": A,
        "startTime": dateTimeUnix,
        "casesPerClass": casesPerClass,
        "grade of Harmonisch":grade
    }

    # Ausgabe der Informationen
    if outputVariables is True:
        print("____________________________________________")
        print("----------------INFORMATION:----------------")
        print("____________________________________________")
        print("Sampling Frequency:      %s Samples/s" % samplingFrequency)
        print("Cycles:                  %s" % cycles)
        print("Netzfrequenz:            %s Hz" % f)
        print("Amplitude des Netzes:    %s V" % A)
        print("Fehlerfälle pro Klasse:  %s" % casesPerClass)
        print("Grade of Harmonisch:  %s" % grade)
        print("____________________________________________")
        print("____________________________________________")
        print("----------------Klassenname:----------------")

    # Liste zum Speichern, hinzufügen wesentlich schneller als Array mit ständiger Erweiterung
    feature = list()
    # Liste der Dieferenz zwischen Sinwave
    delta_list = list()
    # Dictionary of Störung
    dict = {}

    # Normal
    ran = np.random.random(casesPerClass) / 5 - 0.1

    for i in range(0, casesPerClass):
        IntensityOfDisturbance = ran[i]
        StartTimeOfDisturbance = 0
        EndTimeOfDisturbance = cycles * 1 / f
        feature.append(mysinwave(A, f, samplingFrequency, cycles, IntensityOfDisturbance, StartTimeOfDisturbance,
                                 EndTimeOfDisturbance))

    stand = 1
    # Spannungseinbruch
    if outputVariables is True:
        print('Spannungseinbruch ...')
    ranIntensity = np.random.random(casesPerClass) * 0.8 - 0.9
    ranDuration = np.random.random(casesPerClass) * 8 + 1
    for i in range((1 * casesPerClass), (2 * casesPerClass)):
        IntensityOfDisturbance = ranIntensity[i - casesPerClass]
        StartTimeOfDisturbance = 1 * 1 / f
        EndTimeOfDisturbance = (1 + ranDuration[i - casesPerClass]) / f
        feature.append(mysinwave(A, f, samplingFrequency, cycles, IntensityOfDisturbance, StartTimeOfDisturbance,
                                 EndTimeOfDisturbance))

        delta_list.append(mysinwave(A, f, samplingFrequency, cycles, IntensityOfDisturbance, StartTimeOfDisturbance,
                                 EndTimeOfDisturbance)[0:len(x)] - sinwave)

    stand = 2
    dict['Spannungseinbruch'] = stand
    # Ueberspannung
    if outputVariables is True:
        print('Überspannung ...')
    ranIntensity = np.random.random(casesPerClass) * 0.7 + 0.1
    ranDuration = np.random.random(casesPerClass) * 8 + 1
    for i in range((2 * casesPerClass), (3 * casesPerClass)):
        IntensityOfDisturbance = ranIntensity[i - (2 * casesPerClass)]
        StartTimeOfDisturbance = 1 * 1 / f
        EndTimeOfDisturbance = (1 + ranDuration[i - (2 * casesPerClass)]) / f
        feature.append(mysinwave(A, f, samplingFrequency, cycles, IntensityOfDisturbance, StartTimeOfDisturbance,
                                 EndTimeOfDisturbance))

        delta_list.append(mysinwave(A, f, samplingFrequency, cycles, IntensityOfDisturbance, StartTimeOfDisturbance,
                                 EndTimeOfDisturbance)[0:len(x)] - sinwave)

    stand = 3
    dict['Überspannung'] = stand
    # Spannungsunterbrechung
    if outputVariables is True:
        print('Spannungsunterbrechung ...')
    ranIntensity = np.random.random(casesPerClass) * 0.1 - 1
    ranDuration = np.random.random(casesPerClass) * 8 + 1
    for i in range((3 * casesPerClass), (4 * casesPerClass)):
        IntensityOfDisturbance = ranIntensity[i - (3 * casesPerClass)]
        StartTimeOfDisturbance = 1 * 1 / f
        EndTimeOfDisturbance = (1 + ranDuration[i - (3 * casesPerClass)]) / f
        feature.append(mysinwave(A, f, samplingFrequency, cycles, IntensityOfDisturbance, StartTimeOfDisturbance,
                                 EndTimeOfDisturbance))

        delta_list.append(mysinwave(A, f, samplingFrequency, cycles, IntensityOfDisturbance, StartTimeOfDisturbance,
                                    EndTimeOfDisturbance)[0:len(x)] - sinwave)

    stand = 4
    dict['Spannungsunterbrechung'] = stand
    # Harmonische
    if outputVariables is True:
        print('Harmonische ...')
    for i in range((4 * casesPerClass), (5 * casesPerClass)):

        feature.append(myharmonic(A, grade, f, samplingFrequency, cycles))

        delta_list.append(myharmonic(A, grade, f, samplingFrequency, cycles)[0:len(x)] - sinwave)

    stand = 5
    dict['Harmonische'] = stand
    # Spannungseinbruch + Harmonische
    if outputVariables is True:
        print('Spannungseinbruch + Harmonische ...')
    ranIntensity = np.random.random(casesPerClass) * 0.8 - 0.9
    ranDuration = np.random.random(casesPerClass) * 8 + 1
    for i in range((5 * casesPerClass), (6 * casesPerClass)):

        StartTimeOfDisturbance = 1 * 1 / f#没问题
        EndTimeOfDisturbance = (1 + ranDuration[i - (5 * casesPerClass)]) / f#没问题
        IntensityOfDisturbance = ranIntensity[i - (5 * casesPerClass)]#没问题
        feature.append(mysinwaveWithHarmonic(A, f, samplingFrequency, cycles,
                                             IntensityOfDisturbance, StartTimeOfDisturbance, EndTimeOfDisturbance,
                                             grade))

        delta_list.append(mysinwaveWithHarmonic(A, f, samplingFrequency, cycles,
                                             IntensityOfDisturbance, StartTimeOfDisturbance, EndTimeOfDisturbance,
                                             grade)[0:len(x)] - sinwave)

    stand = 6
    dict['Spannungseinbruch + Harmonische'] = stand
    # Ueberspannung + Harmonische
    if outputVariables is True:
        print('Überspannung + Harmonische ...')
    ranIntensity = np.random.random(casesPerClass) * 0.7 + 0.1
    ranDuration = np.random.random(casesPerClass) * 8 + 1
    for i in range((6 * casesPerClass), (7 * casesPerClass)):

        StartTimeOfDisturbance = 1 * 1 / f
        EndTimeOfDisturbance = (1 + ranDuration[i - (6 * casesPerClass)]) / f
        IntensityOfDisturbance = ranIntensity[i - (6 * casesPerClass)]
        feature.append(mysinwaveWithHarmonic(A, f, samplingFrequency, cycles,
                                             IntensityOfDisturbance, StartTimeOfDisturbance, EndTimeOfDisturbance,
                                             grade))

        delta_list.append(mysinwaveWithHarmonic(A, f, samplingFrequency, cycles,
                                                IntensityOfDisturbance, StartTimeOfDisturbance, EndTimeOfDisturbance,
                                                grade)[0:len(x)] - sinwave)

    stand = 7
    dict['Überspannung + Harmonische'] = stand
    # Spannungsunterbrechung + Harmonische
    if outputVariables is True:
        print('Spannungsunterbrechung + Harmonische ...')
    ranIntensity = np.random.random(casesPerClass) * 0.1 - 1
    ranDuration = np.random.random(casesPerClass) * 8 + 1

    for i in range((7 * casesPerClass), (8 * casesPerClass)):

        StartTimeOfDisturbance = 1 * 1 / f
        EndTimeOfDisturbance = (1 + ranDuration[i - (7 * casesPerClass)]) / f
        IntensityOfDisturbance = ranIntensity[i - (7 * casesPerClass)]
        feature.append(mysinwaveWithHarmonic(A, f, samplingFrequency, cycles,
                                             IntensityOfDisturbance, StartTimeOfDisturbance, EndTimeOfDisturbance,
                                             grade))

        delta_list.append(mysinwaveWithHarmonic(A, f, samplingFrequency, cycles,
                                                IntensityOfDisturbance, StartTimeOfDisturbance, EndTimeOfDisturbance,
                                                grade)[0:len(x)] - sinwave)

    stand = 8
    dict['Spannungsunterbrechung + Harmonische'] = stand
    # Flicker
    if outputVariables is True:
        print('Flicker ...')
    IntensityOfDisturbance = 0.0
    StartTimeOfDisturbance = 1 * 1 / f
    EndTimeOfDisturbance = 8 * 1 / f
    randIntensityOfFlicker = np.random.random(casesPerClass) * 0.1 + 0.1
    randFlickerFrequency = np.random.random(casesPerClass) * 0.1 + 0.1
    for i in range((8 * casesPerClass), (9 * casesPerClass)):
        IntensityOfFlicker = randFlickerFrequency[i - (8 * casesPerClass)]
        FlickerFrequency = randFlickerFrequency[i - (8 * casesPerClass)]

        feature.append(myflicker(A, f, samplingFrequency, cycles, IntensityOfDisturbance,
                                 StartTimeOfDisturbance, EndTimeOfDisturbance, IntensityOfFlicker,
                                 FlickerFrequency))

        delta_list.append(myflicker(A, f, samplingFrequency, cycles, IntensityOfDisturbance,
                                 StartTimeOfDisturbance, EndTimeOfDisturbance, IntensityOfFlicker,
                                 FlickerFrequency)[0:len(x)] - sinwave)

    stand = 9
    dict['Flicker'] = stand
    # Oszillierende Transiente
    if outputVariables is True:
        print('Oszillierende Transienten ...')
    randalpha = np.random.random(casesPerClass) * 0.7 + 0.1
    randtf = np.random.random(casesPerClass) * 600 + 300
    randduration = np.random.random(casesPerClass) * 2.5 + 0.5
    randtau = np.random.random(casesPerClass) * 0.0320 + 0.008
    for i in range((9 * casesPerClass), (10 * casesPerClass)):
        alpha = randalpha[i - (9 * casesPerClass)]
        TransientFrequency = randtf[i - (9 * casesPerClass)]
        StartTimeOfDisturbance = 1 * 1 / f
        EndTimeOfDisturbance = (1 + randduration[i - (9 * casesPerClass)]) / f
        tau = randtau[i - (9 * casesPerClass)]
        delta = 0.001

        feature.append(myOscillatorytransient(A, alpha, TransientFrequency,
                                              StartTimeOfDisturbance, EndTimeOfDisturbance, tau, f,
                                              samplingFrequency, cycles, delta))

        delta_list.append(myOscillatorytransient(A, alpha, TransientFrequency,
                                              StartTimeOfDisturbance, EndTimeOfDisturbance, tau, f,
                                              samplingFrequency, cycles, delta)[0:len(x)] - sinwave)

    stand = 10
    dict['Oszillierende Transienten'] = stand
    # Impulsive Transiente
    if outputVariables is True:
        print('Impulsive Transienten ...')
    randalpha = np.random.random(casesPerClass) * 0.7 + 0.1
    randrise = np.random.random(casesPerClass) * (1 * math.pow(10, -4) - 4 * math.pow(10, -6)) + (4 * math.pow(10, -6))
    randduration = np.random.random(casesPerClass) * (0.01 - 4 * math.pow(10, -4)) + (4 * math.pow(10, -4))
    for i in range((10 * casesPerClass), (11 * casesPerClass)):
        alpha = randalpha[i - (10 * casesPerClass)]
        StartTimeOfDisturbance = 1 * 1 / f
        EndTimeOfDisturbance = 4 / f
        risetime = randrise[i - (10 * casesPerClass)]
        duration = randduration[i - (10 * casesPerClass)]
        delta = 0.1

        feature.append(myimpulsiveTransient(A, alpha, StartTimeOfDisturbance,
                                            EndTimeOfDisturbance, risetime, duration,
                                            f, samplingFrequency, cycles, delta))

        delta_list.append(myimpulsiveTransient(A, alpha, StartTimeOfDisturbance,
                                            EndTimeOfDisturbance, risetime, duration,
                                            f, samplingFrequency, cycles, delta)[0:len(x)] - sinwave)

    stand = 11
    dict['Impulsive Transienten'] = stand
    # Periodischer Notch
    if outputVariables is True:
        print('Periodischer Notch ...')
    pos = -1
    randIntensity = np.random.random(casesPerClass) * 0.3 + 0.1
    randDuration = np.random.random(casesPerClass) * (0.05 * 1 / f - 0.01 * 1 / f) + 0.01 * 1 / f
    for i in range((11 * casesPerClass), (12 * casesPerClass)):
        intensity = randIntensity[i - (11 * casesPerClass)]
        t1 = 0.2 * 1 / f
        duration = randDuration[i - (11 * casesPerClass)]
        t2 = t1 + duration
        feature.append(myNotch(A, f, samplingFrequency, cycles, intensity, t1, t2, pos))

        delta_list.append(myNotch(A, f, samplingFrequency, cycles, intensity, t1, t2, pos)[0:len(x)] - sinwave)

    stand = 12
    dict['Periodischer Notch'] = stand
    # Periodischer Spike
    if outputVariables is True:
        print('Periodischer Spike ...')
    pos = 1
    randIntensity = np.random.random(casesPerClass) * 0.3 + 0.1
    randDuration = np.random.random(casesPerClass) * (0.05 * 1 / f - 0.01 * 1 / f) + 0.01 * 1 / f
    for i in range((12 * casesPerClass), (13 * casesPerClass)):
        intensity = randIntensity[i - (12 * casesPerClass)]
        t1 = 0.2 * 1 / f
        duration = randDuration[i - (12 * casesPerClass)]
        t2 = t1 + duration
        feature.append(myNotch(A, f, samplingFrequency, cycles, intensity, t1, t2, pos))

        delta_list.append(myNotch(A, f, samplingFrequency, cycles, intensity, t1, t2, pos)[0:len(x)] - sinwave)

    stand = 13
    dict['Periodischer Spike'] = stand
    # Flicker + Harmonische
    if outputVariables is True:
        print('Flicker + Harmonische ...')

    randIntensityOfFlicker = np.random.random(casesPerClass) * 0.1 + 0.1
    randFlickerFrequency = np.random.random(casesPerClass) * 0.1 + 0.1
    for i in range((13 * casesPerClass), (14 * casesPerClass)):

        IntensityOfFlicker = randIntensityOfFlicker[i - (13 * casesPerClass)]
        FlickerFrequency = randFlickerFrequency[i - (13 * casesPerClass)]
        feature.append(myflickerWithHarmonic(A, grade, f,
                                             samplingFrequency, cycles, IntensityOfFlicker, FlickerFrequency))

        delta_list.append(myflickerWithHarmonic(A, grade, f,
                                             samplingFrequency, cycles, IntensityOfFlicker, FlickerFrequency)[0:len(x)] - sinwave)

    stand = 14
    dict['Flicker + Harmonische'] = stand
    # Spannungseinbruch + Flicker
    if outputVariables is True:
        print('Spannungseinbruch + Flicker ...')
    ranIntensity = np.random.random(casesPerClass) * 0.8 - 0.9
    ranDuration = np.random.random(casesPerClass) * 8 + 1
    randIntensityOfFlicker = np.random.random(casesPerClass) * 0.1 + 0.1
    randFlickerFrequency = np.random.random(casesPerClass) * 0.1 + 0.1
    for i in range((14 * casesPerClass), (15 * casesPerClass)):
        IntensityOfFlicker = randFlickerFrequency[i - (14 * casesPerClass)]
        FlickerFrequency = randFlickerFrequency[i - (14 * casesPerClass)]

        IntensityOfDisturbance = ranIntensity[i - (14 * casesPerClass)]
        StartTimeOfDisturbance = 1 * 1 / f
        EndTimeOfDisturbance = (1 + ranDuration[i - (14 * casesPerClass)]) / f

        feature.append(myflicker(A, f, samplingFrequency, cycles, IntensityOfDisturbance,
                                 StartTimeOfDisturbance, EndTimeOfDisturbance, IntensityOfFlicker, FlickerFrequency))

        delta_list.append(myflicker(A, f, samplingFrequency, cycles, IntensityOfDisturbance,
                                 StartTimeOfDisturbance, EndTimeOfDisturbance, IntensityOfFlicker, FlickerFrequency)[0:len(x)] - sinwave)

    stand = 15
    dict['Spannungseinbruch + Flicker'] = stand
    # Ueberspannung + Flicker
    if outputVariables is True:
        print('Überspannung + Flicker ...')
    ranIntensity = np.random.random(casesPerClass) * 0.7 + 0.1
    ranDuration = np.random.random(casesPerClass) * 8 + 1
    randIntensityOfFlicker = np.random.random(casesPerClass) * 0.1 + 0.1
    randFlickerFrequency = np.random.random(casesPerClass) * 0.1 + 0.1
    for i in range((15 * casesPerClass), (16 * casesPerClass)):
        IntensityOfFlicker = randFlickerFrequency[i - (15 * casesPerClass)]
        FlickerFrequency = randFlickerFrequency[i - (15 * casesPerClass)]

        IntensityOfDisturbance = ranIntensity[i - (15 * casesPerClass)]
        StartTimeOfDisturbance = 1 * 1 / f
        EndTimeOfDisturbance = (1 + ranDuration[i - (15 * casesPerClass)]) / f

        feature.append(myflicker(A, f, samplingFrequency, cycles, IntensityOfDisturbance,
                                 StartTimeOfDisturbance, EndTimeOfDisturbance, IntensityOfFlicker, FlickerFrequency))

        delta_list.append(myflicker(A, f, samplingFrequency, cycles, IntensityOfDisturbance,
                                    StartTimeOfDisturbance, EndTimeOfDisturbance, IntensityOfFlicker,
                                    FlickerFrequency)[0:len(x)] - sinwave)

    stand = 16
    dict['Überspannung + Flicker'] = stand
    # Spannungsunterbrechung + Flicker
    if outputVariables is True:
        print('Spannungsunterbrechung + Flicker ...')
    ranIntensity = np.random.random(casesPerClass) * 0.1 - 1
    ranDuration = np.random.random(casesPerClass) * 8 + 1
    randIntensityOfFlicker = np.random.random(casesPerClass) * 0.1 + 0.1
    randFlickerFrequency = np.random.random(casesPerClass) * 0.1 + 0.1
    for i in range((16 * casesPerClass), (17 * casesPerClass)):
        IntensityOfFlicker = randFlickerFrequency[i - (16 * casesPerClass)]
        FlickerFrequency = randFlickerFrequency[i - (16 * casesPerClass)]
        IntensityOfDisturbance = ranIntensity[i - (16 * casesPerClass)]
        StartTimeOfDisturbance = 1 * 1 / f
        EndTimeOfDisturbance = (1 + ranDuration[i - (16 * casesPerClass)]) / f
        feature.append(myflicker(A, f, samplingFrequency, cycles, IntensityOfDisturbance,
                                 StartTimeOfDisturbance, EndTimeOfDisturbance, IntensityOfFlicker, FlickerFrequency))

        delta_list.append(myflicker(A, f, samplingFrequency, cycles, IntensityOfDisturbance,
                                    StartTimeOfDisturbance, EndTimeOfDisturbance, IntensityOfFlicker,
                                    FlickerFrequency)[0:len(x)] - sinwave)

    stand = 17
    dict['Spannungsunterbrechung + Flicker'] = stand

    # if outputVariables is True:
    #     print('Kombinationsfälle')
    #
    # a = creatKombination(A,delta_list,kombination_list,dict)
    # for i in a:
    #     feature.append(i)
    #
    # stand = 18

    # Save the data
    print("____________________________________________")
    print('SAVE THE DATA ...')
    print("____________________________________________")

    # alle Informationen in Datei - Name
    filename = 'simulation_daten(sf-%s_f-%s_cycles-%s_cases-%s).csv' % (samplingFrequency, f, cycles, casesPerClass)

    # Metadaten nur an Ende der Liste als Dictionary
    # feature.append(informationDict)

    path = "Rohdaten\\Simulation_Daten(Wellenform)\\"
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    a = pd.DataFrame(feature)
    path= path+filename
    a.to_csv(path,index=False)





    # plotData(f, samplingFrequency, cycles, A, casesPerClass,delta_list,kombination_list,dict)
    #
    # # Information Ende des Vorgangs
    # print('Berechnungen abgeschlossen!')
    # print("____________________________________________")
    # print("____________________________________________")
