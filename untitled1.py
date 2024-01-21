# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:36:31 2023

@author: Michel Gordillo
"""

import os 

isr_bins_file = os.path.normpath('E:/Documentos/Thesis - Master/Master/ISR_bins.txt')

with open(isr_bins_file, 'r') as f:
    lines = f.readlines()

lines[21] = '10000000.0'

lim_inf = [float(line.rstrip()) for line in lines[0:11]]
lim_sup = [float(line.rstrip()) for line in lines[11:22]]
cuota = [float(line.rstrip()) for line in lines[22:33]]
percentage = [float(line.rstrip()) for line in lines[33:]]

base_salary = 28874.84


for i, (minimum, maximum) in enumerate(zip(lim_inf,lim_sup)):

    if base_salary >= minimum and base_salary <= maximum:
        nivel = i
        print('\nNivel: {}°: \nLímite inferior: {}\nLímite superior: {}\nCuota: {}\nPorcentaje: {}\n'.format(nivel,minimum,maximum,cuota[nivel],percentage[nivel]))
        


diferencia = base_salary - lim_inf[nivel]
tasa = diferencia*percentage[nivel]/100
isr_determined = tasa + cuota[nivel]
ingreso_bruto = base_salary - isr_determined
vales_despensa = 0.1*base_salary

print('ISR: {}\nSueldo Bruto: {}\nVales Depensa: {}'.format(isr_determined,ingreso_bruto,vales_despensa))



p_bono_desemp = 0.08

tasa_bonos = 0.22

bono_desemp = 12*base_salary*p_bono_desemp*(1 - tasa_bonos)

percepcion_anual = (ingreso_bruto + vales_despensa)*12 + bono_desemp
