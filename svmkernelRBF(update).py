from textwrap import indent
from typing import List
from numpy.core.defchararray import index
from numpy.core.fromnumeric import argmax
from numpy.core.records import array
import pandas as pd
import numpy as np
import math

from svmkernelRBF import bias, nilaiE1, wnegatif

# Tahap 1 kernel RBF (Data latih) ==============================================


def kernelrbf(datatrain):
    lendata = 1
    lendata2 = len(datatrain)
    rowkernel = []

    jumlah = []
    for i, val in enumerate(datatrain):
        # print(val, "i")
        for j, val1 in enumerate(datatrain):
            kernel = 0
            # print(val1, "j")
            for n in range(30):
                kernel += (val[n] - val1[n]) ** 2
            jumlah = kernel/2
            hasilrbf = math.exp(-1 * jumlah)
            rowkernel.append(hasilrbf)
    # print(rowkernel)
    return rowkernel

# ================= Tahap 2 sequential trainig (Data latih) ======================
# level 1 : -------------------------------------


def matrixhessian1(krnl):
    hessian1 = []
    for i, val in enumerate(krnl):
        hasilmatrix = val + 0.25
        hessian1.append(hasilmatrix)
    # print(hessian1)
    return hessian1


def errorrate1(mh1, kelaslvl1):
    error1 = []
    for i, val in enumerate(mh1):
        er1 = val * 0.008
        error1.append(er1)
    # print(erro1)

    fixer1 = np.reshape(error1, (30, 30))
    # print(fixer1)

    er1 = [a * b for a, b in zip(fixer1, kelaslvl1)]
    #print("kali", er1)

    E1 = []
    for i, val in enumerate(er1):
        # print(val, "i")
        total = 0
        # print(val1, "j")
        for n in range(30):
            total += (val[n])
        jumlah = total
        E1.append(jumlah)
    # print(E1)
    return E1


def deltaalfa1(E1):
    dalfa1 = []
    for i, val in enumerate(E1):
        hsildalfa1 = 0.008 * (1-val)
        dalfa1.append(hsildalfa1)
    # print(dalfa1)
    return dalfa1


def alfalvl1(da1):
    alfa1 = []
    for i, val in enumerate(da1):
        total = val + 0.008
        alfa1.append(total)
    # print(alfa1)
    return alfa1

# level 2 : ------------------------------------------------


def errorrate2(mh1, kelaslvl2):
    error1 = []
    for i, val in enumerate(mh1):
        er1 = val * 0.008
        error1.append(er1)
    # print(error1)

    fixer1 = np.reshape(error1, (30, 30))
    fixer1 = np.delete(fixer1, [23, 24, 25, 26, 27, 28, 29], axis=1)
    fixer1 = np.delete(fixer1, [23, 24, 25, 26, 27, 28, 29], axis=0)
    # rint(fixer1)

    er1 = [a * b for a, b in zip(fixer1, kelaslvl2)]
    #print("kali", er1)

    E1 = []
    for i, val in enumerate(er1):
        # print(val, "i")
        total = 0
        # print(val1, "j")
        for n in range(23):
            total += (val[n])
        jumlah = total
        E1.append(jumlah)
    # print(E1)
    return E1


def deltaalfa2(E2):
    dalfa1 = []
    for i, val in enumerate(E2):
        hsildalfa1 = 0.008 * (1-val)
        dalfa1.append(hsildalfa1)
    # print(dalfa1)
    return dalfa1


def alfalvl2(da2):
    alfa1 = []
    for i, val in enumerate(da2):
        total = val + 0.008
        alfa1.append(total)
    # print(alfa1)
    return alfa1

# level 3 : ------------------------------------------------


def errorrate3(mh1, kelaslvl3):
    error1 = []
    for i, val in enumerate(mh1):
        er1 = val * 0.008
        error1.append(er1)
    # print(error1)

    fixer1 = np.reshape(error1, (30, 30))
    fixer1 = np.delete(fixer1, [16, 17, 18, 19, 20,
                                21, 22, 23, 24, 25, 26, 27, 28, 29], axis=1)
    fixer1 = np.delete(fixer1, [16, 17, 18, 19, 20,
                                21, 22, 23, 24, 25, 26, 27, 28, 29], axis=0)
    # print(fixer1)

    er1 = [a * b for a, b in zip(fixer1, kelaslvl3)]
    #print("kali", er1)

    E1 = []
    for i, val in enumerate(er1):
        # print(val, "i")
        total = 0
        # print(val1, "j")
        for n in range(16):
            total += (val[n])
        jumlah = total
        E1.append(jumlah)
    # print(E1)
    return E1


def deltaalfa3(E3):
    dalfa1 = []
    for i, val in enumerate(E3):
        hsildalfa1 = 0.008 * (1-val)
        dalfa1.append(hsildalfa1)
    # print(dalfa1)
    return dalfa1


def alfalvl3(da3):
    alfa1 = []
    for i, val in enumerate(da3):
        total = val + 0.008
        alfa1.append(total)
    # print(alfa1)
    return alfa1
# ====================== Weight Level 1 ==========================================


def weightpos1(a1, krnl, kelaslvl1):

    fixkrnl = np.reshape(krnl, (30, 30))
    # print(fixkrnl)
    # K(xxi)
    apos = np.array(a1)
    index = (23, 24, 25, 26, 27, 28, 29)
    alfamax1 = np.delete(apos, index)
    # print(alfamax1)
    #print('nilai maks alfa kls pos : ', max(alfamax1))
    #print("list1[G16] : ", fixkrnl[15])
    klspos1 = fixkrnl[15]
    klslvl1 = kelaslvl1[0]
    # print(klspos1)

    hasil = [a * b for a, b in zip(a1, klspos1)]
    hasil2 = [c * d for c, d in zip(hasil, klslvl1)]
    #print("kali", hasil2)

    sum = 0
    for i in range(0, len(hasil2)):
        sum = sum + hasil2[i]
    wp1 = sum
    # print(wp1)
    return wp1


def weightneg1(a1, krnl, kelaslvl1):

    fixkrnl = np.reshape(krnl, (30, 30))
    # print(fixkrnl)

    aneg = np.array(a1)
    index = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
    alfamax1 = np.delete(aneg, index)
    # print(alfamax1)
    #print('nilai maks alfa kls neg : ', max(alfamax1))
    #print("list1[G30] : ", fixkrnl[29])
    klsneg1 = fixkrnl[29]
    klslvl1 = kelaslvl1[0]
    # print(klsneg1)
    hasil = [a * b for a, b in zip(a1, klsneg1)]
    hasil2 = [c * d for c, d in zip(hasil, klslvl1)]
    #print("kali", hasil2)

    sum = 0
    for i in range(0, len(hasil2)):
        sum = sum + hasil2[i]
    wn1 = sum
    # print(wn1)
    return wn1

# ====================== Weight Level 2 ==========================================


def weightpos2(a2, krnl, kelaslvl2):

    fixkrnl = np.reshape(krnl, (30, 30))
    # print(fixkrnl)
    # K(xxi)
    apos = np.array(a2)
    index = (16, 17, 18, 19, 20, 21, 22)
    alfamax1 = np.delete(apos, index)
    # print(alfamax1)
    #print('nilai maks alfa kls pos level 2 : ', max(alfamax1))
    #print("list1[G16] : ", fixkrnl[15])
    klspos1 = fixkrnl[14]
    klslvl1 = kelaslvl2[0]
    # print(klspos1)

    hasil = [a * b for a, b in zip(a2, klspos1)]
    hasil2 = [c * d for c, d in zip(hasil, klslvl1)]
    #print("kali", hasil2)

    sum = 0
    for i in range(0, len(hasil2)):
        sum = sum + hasil2[i]
    wp1 = sum
    # print(wp1)
    return wp1


def weightneg2(a2, krnl, kelaslvl2):

    fixkrnl = np.reshape(krnl, (30, 30))
    # print(fixkrnl)

    aneg = np.array(a2)
    index = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
             13, 14, 15)
    alfamax1 = np.delete(aneg, index)
    # print(alfamax1)
    #print('nilai maks alfa kls neg level 2 : ', max(alfamax1))
    #print("list1[G30] : ", fixkrnl[29])
    klsneg1 = fixkrnl[18]
    klslvl1 = kelaslvl2[0]
    # print(klsneg1)
    hasil = [a * b for a, b in zip(a2, klsneg1)]
    hasil2 = [c * d for c, d in zip(hasil, klslvl1)]
    #print("kali", hasil2)

    sum = 0
    for i in range(0, len(hasil2)):
        sum = sum + hasil2[i]
    wn1 = sum
    # print(wn1)
    return wn1

# ====================== Weight Level 3 ==========================================


def weightpos3(a3, krnl, kelaslvl3):

    fixkrnl = np.reshape(krnl, (30, 30))
    # print(fixkrnl)
    # K(xxi)
    apos = np.array(a3)
    index = (8, 9, 10, 11, 12, 13, 14, 15)
    alfamax1 = np.delete(apos, index)
    # print(alfamax1)
    #print('nilai maks alfa kls pos level 3 : ', max(alfamax1))
    #print("list1[G7] : ", fixkrnl[6])
    klspos1 = fixkrnl[6]
    klslvl1 = kelaslvl3[0]
    # print(klspos1)

    hasil = [a * b for a, b in zip(a3, klspos1)]
    hasil2 = [c * d for c, d in zip(hasil, klslvl1)]
    #print("kali", hasil2)

    sum = 0
    for i in range(0, len(hasil2)):
        sum = sum + hasil2[i]
    wp1 = sum
    # print(wp1)
    return wp1


def weightneg3(a3, krnl, kelaslvl3):

    fixkrnl = np.reshape(krnl, (30, 30))
    # print(fixkrnl)

    aneg = np.array(a3)
    index = (0, 1, 2, 3, 4, 5, 6, 7)
    alfamax1 = np.delete(aneg, index)
    # print(alfamax1)
    #print('nilai maks alfa kls neg level 3 : ', max(alfamax1))
    #print("list1[G14] : ", fixkrnl[13])
    klsneg1 = fixkrnl[13]
    klslvl1 = kelaslvl3[0]
    # print(klsneg1)
    hasil = [a * b for a, b in zip(a3, klsneg1)]
    hasil2 = [c * d for c, d in zip(hasil, klslvl1)]
    #print("kali", hasil2)

    sum = 0
    for i in range(0, len(hasil2)):
        sum = sum + hasil2[i]
    wn1 = sum
    # print(wn1)
    return wn1

# ========================= bias ==========================================


def bias1(wpos1, wneg1):

    jumlahbias = -0.5 * (wpos1 + wneg1)
    # print(jumlahbias)
    return jumlahbias


def bias2(wpos2, wneg2):

    jumlahbias = -0.5 * (wpos2 + wneg2)
    # print(jumlahbias)
    return jumlahbias


def bias3(wpos3, wneg3):

    jumlahbias = -0.5 * (wpos3 + wneg3)
    # print(jumlahbias)
    return jumlahbias

#==============================Testing==================================#


def kerneltesting(datatrain, datatest):
    lendata = 1
    lendata2 = len(datatrain)
    rowkernel = []

    jumlah = []
    for i, val in enumerate(datatest):
        # print(val, "i")
        for j, val1 in enumerate(datatrain):
            kernel = 0
            # print(val1, "j")
            for n in range(30):
                kernel += (val[n] - val1[n]) ** 2
            jumlah = kernel/2
            hasilrbf = math.exp(-1 * jumlah)
            rowkernel.append(hasilrbf)
    # print(rowkernel)
    return rowkernel

# --------------------- WT 1-------------------------


def weightest1(krnluji, a1, kelaslvl1):

    klslvl1 = kelaslvl1[0]
    hasil = [a * b for a, b in zip(a1, krnluji)]
    hasil2 = [c*d for c, d in zip(hasil, klslvl1)]
    #print("Level1", hasil2)

    sum = 0
    for i in range(0, len(hasil2)):
        sum = sum + hasil2[i]
    wtest1 = sum
    # print(wtest1)
    return wtest1

# --------------------- WT 2-------------------------


def weightest2(krnluji, a2, kelaslvl2):

    klslvl1 = kelaslvl2[0]
    hasil = [a * b for a, b in zip(a2, krnluji)]
    hasil2 = [c*d for c, d in zip(hasil, klslvl1)]
    #print("Level1", hasil2)

    sum = 0
    for i in range(0, len(hasil2)):
        sum = sum + hasil2[i]
    wtest1 = sum
    # print(wtest1)
    return wtest1
# --------------------- WT 3-------------------------


def weightest3(krnluji, a3, kelaslvl3):

    klslvl1 = kelaslvl3[0]
    hasil = [a * b for a, b in zip(a3, krnluji)]
    hasil2 = [c*d for c, d in zip(hasil, klslvl1)]
    #print("Level1", hasil2)

    sum = 0
    for i in range(0, len(hasil2)):
        sum = sum + hasil2[i]
    wtest1 = sum
    # print(wtest1)
    return wtest1

#==============================Fungsi==================================#


def fungsi1(wtest1, nbias1):
    fxlvl1 = wtest1 + nbias1
    # print(fxlvl1)
    return fxlvl1


def fungsi2(wtest2, nbias2):
    fxlvl1 = wtest2 + nbias2
    # print(fxlvl1)
    return fxlvl1


def fungsi3(wtest3, nbias3):
    fxlvl1 = wtest3 + nbias3
    # print(fxlvl1)
    return fxlvl1


def klasifikasi(fungsi1, fungsi2, fungsi3):
    if fungsi1 < 0:
        kls = 'Sangat Tinggi'
    elif fungsi2 < 0:
        kls = 'Tinggi'
    elif fungsi3 < 0:
        kls = 'Sedang'
    else:
        kls = 'Rendah'
    print(kls)



    