import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from hmmlearn import hmm


def nb_egitim(egitim_veri):
    onceki = egitim_veri.groupby(["letter"]).count()["id"] / len(egitim_veri)
    on_islem = egitim_veri.groupby(['letter']).sum().iloc[:, 5:134] / egitim_veri.groupby(["letter"]).count().iloc[:,5:134]
    return onceki, on_islem


def nb_dogruluk(test_veri, egitim_parametre):
    table = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
             "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
             "z": 25}
    tahmin_sonuc = np.zeros(len(test_veri))
    id = 0
    count = 0
    for index, row in test_veri.iterrows():
        tahmin_sonuc[id] = nb_tahmin(row, egitim_parametre)
        if tahmin_sonuc[id] == table.get(row["letter"]):
            count += 1
        id +=1
    doğruluk = count / len(test_veri)
    return doğruluk, tahmin_sonuc


def nb_tahmin(data, egitim_parametre):
    onceki, on_islem = egitim_parametre
    result_pro = np.zeros(26)
    for i in range(26):
        result_pro[i] = np.sum([np.log(on_islem.iloc[i, j]) if data.iloc[6 + j] == 1 else np.log(1 - on_islem.iloc[i, j]) for j in range(128)]) + np.log(onceki[i])
    return np.argmax(result_pro)


def hmm_egitim(hmm_egitim_veri):
    table = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
             "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
             "z": 25}
    transfer_matris = np.zeros((26, 26))
    baslatproblem = np.zeros(26)
    yayılımproblem = np.zeros((26, 26))
    for index, row in hmm_egitim_veri.iterrows():
        baslatproblem[table.get(row["letter"])] += 1
        yayılımproblem[table.get(row["letter"]), table.get(row["bayes"])] += 1
        if row["next_id"] == -1:
            continue
        next_letter = hmm_egitim_veri.loc[row["next_id"] - 1, "letter"]
        transfer_matris[table.get(row["letter"]), table.get(next_letter)] += 1

    transfer_matris = transfer_matris / np.sum(transfer_matris, axis=1).reshape(-1, 1)
    yayılımproblem = yayılımproblem / np.sum(yayılımproblem, axis=1).reshape(-1, 1)
    baslatproblem = baslatproblem / np.sum(baslatproblem)
    return transfer_matris, yayılımproblem, baslatproblem


def hmm_accuracy(hmm_test_verisi, hmm_egitim_sonuc):
    table = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
             "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
             "z": 25}
    hmm_model = hmm.MultinomialHMM(n_components=26)

    hmm_model.transmat_, hmm_model.emissionprob_, hmm_model.startprob_ = hmm_egitim_sonuc
    sorgudizi = []
    sonuçdizi = []
    for index, row in hmm_test_verisi.iterrows():
        sorgudizi.append(table.get(hmm_test_verisi.loc[index, "letter"]))
        if (hmm_test_verisi.loc[index, "next_id"] == -1):
            sonuçdizi.extend(hmm_model.predict(np.array(sorgudizi).reshape(-1, 1)))
            sorgudizi = []
    
    doğruluk = 0
    table2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
              "u",
              "v", 'w', "x", "y", "z"]
    beklenen_dizi = []
    hmm_test_data_ite = hmm_test_verisi.iterrows()
    for j in range(len(sonuçdizi)):
        if table2[sonuçdizi[j]] == hmm_test_verisi.loc[next(hmm_test_data_ite)[0], "letter"]:
            doğruluk += 1
        beklenen_dizi.append(table2[sonuçdizi[j]])
    doğruluk /= len(sonuçdizi)
    return doğruluk


def main():
    header = pd.read_csv("letter.names", header=None)
    data = pd.read_csv("letter.data", sep="\s+", names=header.values.reshape(1, -1)[0])

    '''
    10 cross fold  naive bayes 
    '''
    bayes_sonuc = np.zeros(10)
    for i in range(10):
        egitim_veri = data[data["fold"] != i]
        test_veri = data[data["fold"] == i]
        egitim_veri.index = range(len(egitim_veri))
        test_veri.index = range(len(test_veri))
        clf = BernoulliNB()
        clf.fit(egitim_veri.iloc[:, 6:134], egitim_veri.iloc[:, 1])
        bayes_sonuc[i] = clf.score(test_veri.iloc[:, 6:134], test_veri.iloc[:, 1])


    '''
     naive bayes ile hmm'yi birleştir
    '''
    beyess_tahmins = clf.predict(data.iloc[:, 6:134])5
    bayes = pd.DataFrame(beyess_tahmins.reshape(-1, 1), columns=["bayes"])
    hmm_veri = pd.concat([data, bayes], axis=1)
    table = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
             "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
             "z": 25}

    '''
    10 cross fold HMM
    '''
    hmm_sonuc = np.zeros(10)
    for i in range(10):
        hmm_egitim_veri = hmm_veri[hmm_veri["fold"] != i]
        hmm_test_verisi = hmm_veri[hmm_veri["fold"] == i]
        hmm_sonuc[i] = hmm_accuracy(hmm_test_verisi, hmm_egitim(hmm_egitim_veri))

    print("Yanlızca Naive Bayes İçin Sonuç: ")
    print(bayes_sonuc)
    print("Ortalama Sonuç ")
    print(np.average(bayes_sonuc))
    print("HMM ile Birleştirilmiş Naive Bayes İçin Sonuç")
    print(hmm_sonuc)
    print("Ortalama Sonuç: ")
    print(np.average(hmm_sonuc))


if __name__ == '__main__':
    main()
