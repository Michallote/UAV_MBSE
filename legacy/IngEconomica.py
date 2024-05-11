# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:36:31 2023

@author: Michel Gordillo
"""

import datetime
import io

import numpy as np
import pandas as pd
from sklearn import base

ISR_string = """
           LI           LS      Cuota  tasa_excedente
0        0.01       644.58       0.00            1.92
1      644.59      5470.92      12.38            6.40
2     5470.93      9614.66     321.26           10.88
3     9614.67     11176.62     772.10           16.00
4    11176.63     13381.47    1022.01           17.92
5    13381.48     26988.50    1417.12           21.36
6    26988.51     42537.58    4323.58           23.52
7    42537.59     81211.25    7980.73           30.00
8    81211.26    108281.67   19582.83           32.00
9   108281.68    324845.01   28245.36           34.00
10  324845.02  10000000.00  101876.90           35.00
"""

df_ISR = pd.read_csv(io.StringIO(ISR_string), delim_whitespace=True)

lim_inf = list(df_ISR["LI"])
lim_sup = list(df_ISR["LS"])
cuota = list(df_ISR["Cuota"])
percentage = list(df_ISR["tasa_excedente"])

base_salary = 40000.00


def det_niveles(base_salary):
    base_salary = round(base_salary, 2)

    for i, (minimum, maximum) in enumerate(zip(lim_inf, lim_sup)):

        if base_salary >= minimum and base_salary <= maximum:
            nivel = i
            print(
                "\nSueldo Base: {}\nNivel: {}°: \nLímite inferior: {}\nLímite superior: {}\nCuota: {}\nPorcentaje: {}\n".format(
                    base_salary,
                    nivel + 1,
                    minimum,
                    maximum,
                    cuota[nivel],
                    percentage[nivel],
                )
            )
            break

    return nivel


def ISR(base_salary):
    nivel = det_niveles(base_salary)
    diferencia = base_salary - lim_inf[nivel]
    tasa = diferencia * percentage[nivel] / 100
    isr_determined = tasa + cuota[nivel]
    print("Tasa: {:.2f}\nISR: {:.2f}\n".format(tasa, isr_determined))
    return isr_determined


def IMSS(base_salary, prima_vacacional=0.25, dias_vacaciones=12, dias_aguinaldo=15):
    """
    Parameters
    ----------
    base_salary : TYPE
        DESCRIPTION.
    prima_vacacional : TYPE, optional
        DESCRIPTION. The default is 0.6.
    dias_vacaciones : TYPE, optional
        DESCRIPTION. The default is 12.
    dias_aguinaldo : TYPE, optional
        DESCRIPTION. The default is 30.

    Tasas de Retiro:
    # Excedente cuota fija = 0.4
    # Prestaciones en dinero = 0.25
    # Gastos médicos = 0.375
    # Invalidez y vida = 0.625
    # Cesantía y Vejez = 1.125
    """
    salario_diario = base_salary / 30
    factor_prima_vacacional = (
        dias_vacaciones * prima_vacacional / 365
    )  # 60% de los días de vacaciones
    factor_aguinaldo = dias_aguinaldo / 365  # Siempre que escoja 30 días
    factor_integracion_IMSS = 1 + factor_prima_vacacional + factor_aguinaldo
    salario_diario_INTEGRADO_IMSS = salario_diario * factor_integracion_IMSS
    salario_mensual_INTEGRADO_IMSS = salario_diario_INTEGRADO_IMSS * 30
    tarifas = 0.4 + 0.25 + 0.375 + 0.625 + 1.125
    tasas_de_aportacion_retiro = tarifas / 100  # Tasas de aportacion del IMSS al obrero
    cuota_mensual_IMSS = salario_mensual_INTEGRADO_IMSS * tasas_de_aportacion_retiro

    print("IMSS: {:.2f}\n".format(cuota_mensual_IMSS))
    return cuota_mensual_IMSS


def vales_despensa(base_salary, p=0.1, metodo="ISR"):
    """
    p = porcentaje de vales de despensa del salario

    La Ley del Seguro Social establece que todo monto destinado a vales de
    despensa será excluido siempre y cuando no rebase el 40% del valor de UMA.

    La Ley del Seguro Social menciona que, cuando el importe de las prestaciones
    rebase el porcentaje establecido, solamente se integrarán los excedentes al
    salario base de cotización.

    Al empleado se le retendrán impuestos por IMSS e ISR, únicamente por la
    cantidad excedida.

    """

    factor_metodo = {"ISR": 1, "IMSS": 0.4}

    UMA = 103.3
    lim_impuesto = factor_metodo[metodo] * UMA * 30

    monto_vales = base_salary * p

    excedente = monto_vales - lim_impuesto
    return monto_vales, max(excedente, 0)


def fondo_ahorro(base_salary, monto=None, p=0.13):

    monthly_limit = p * base_salary

    if not bool(monto):
        return monthly_limit
    elif monto > monthly_limit:
        return monthly_limit
    else:
        return monto


def bono_anual_desemp(
    base_salary, p_bono_desemp=0.08, p_hacienda=0.25, p_medicare=0.0145, p_seguro=0.062
):
    bono_desemp = 12 * base_salary * p_bono_desemp

    impuestos = (p_hacienda + p_medicare + p_seguro) * bono_desemp

    return bono_desemp, impuestos


def aguinaldo(base_salary, startdate, aguinaldo_date, dias_aguinaldo=15):
    salario_jornada = base_salary / 30

    dias_trabajados = np.busday_count(startdate, aguinaldo_date, weekmask="1111111")

    if dias_trabajados < 365:
        dias_aguinaldo = dias_aguinaldo * dias_trabajados / 365

    monto = salario_jornada * dias_aguinaldo
    return monto


def legacy_calcular_salario_integrado(
    monto_vales: float | None = None, p_vales: float = 0.1
):

    base_salary = 40000
    bono_puntualidad = bono_asistencia = 0.1
    monto_teletrabajo = 133
    dias_aguinaldo = 15
    prima_vacacional = 0.25
    dias_vacaciones = 18
    p_bono_desemp = 0.08
    monto_vales = 2047.0

    if monto_vales is None:
        monto_vales, excedente = vales_despensa(base_salary, p=p_vales)

    isr = ISR(base_salary + excedente)
    cuota_IMSS = IMSS(
        base_salary,
        prima_vacacional=prima_vacacional,
        dias_vacaciones=dias_vacaciones,
        dias_aguinaldo=dias_aguinaldo,
    )
    fondo = fondo_ahorro(base_salary, p=0.03)
    bonos_mensuales = (
        bono_puntualidad * base_salary
        + bono_asistencia * base_salary
        + monto_teletrabajo
    )

    print(
        f"{base_salary=}, {isr=:.2f}, {cuota_IMSS=:.2f}, {monto_vales=:.2f}, {fondo=:.2f}, {bonos_mensuales=:.2f}"
    )

    salario_neto_mensual = (
        base_salary - isr - cuota_IMSS + monto_vales - fondo + bonos_mensuales
    )

    print(f"{salario_neto_mensual=:.2f}")

    salario_anual_integrado = (salario_neto_mensual + 2 * fondo) * 12

    bono_anual, impuestos_bono = bono_anual_desemp(
        base_salary,
        p_bono_desemp=p_bono_desemp,
    )

    salario_anual_integrado += bono_anual - impuestos_bono

    startdate = datetime.date(year=2023, month=3, day=16)
    aguinaldo_date = datetime.date(year=2024, month=12, day=16)

    monto_aguinaldo = aguinaldo(
        base_salary, startdate, aguinaldo_date, dias_aguinaldo=dias_aguinaldo
    )
    monto_vacaciones = dias_vacaciones * prima_vacacional * base_salary / 30

    salario_anual_integrado += monto_aguinaldo + monto_vacaciones


def calcular_salario_integrado(
    base_salary: float,
    bono_puntualidad: float = 0.0,
    bono_asistencia: float = 0.0,
    monto_teletrabajo: float = 0.0,
    dias_aguinaldo: int = 15,
    prima_vacacional: float = 0.25,
    dias_vacaciones: int = 12,
    p_bono_desemp: float = 0.0,
    p_fondo: float = 0.00,
    monto_vales: float | None = None,
    p_vales: float = 0.0,
    extra_mensuales: float = 0.0,
    extra_anuales: float = 0.0,
):

    if monto_vales is None:
        monto_vales, excedente = vales_despensa(base_salary, p=p_vales)
    else:
        excedente = 0

    isr = ISR(base_salary + excedente)
    cuota_IMSS = IMSS(
        base_salary,
        prima_vacacional=prima_vacacional,
        dias_vacaciones=dias_vacaciones,
        dias_aguinaldo=dias_aguinaldo,
    )
    fondo = fondo_ahorro(base_salary, p=p_fondo)
    bonos_mensuales = (
        bono_puntualidad * base_salary
        + bono_asistencia * base_salary
        + monto_teletrabajo
        + extra_mensuales
    )

    print(
        f"{base_salary=}, {isr=:.2f}, {cuota_IMSS=:.2f}, {monto_vales=:.2f}, {fondo=:.2f}, {bonos_mensuales=:.2f}"
    )

    salario_neto_mensual = (
        base_salary - isr - cuota_IMSS + monto_vales - fondo + bonos_mensuales
    )

    print(f"{salario_neto_mensual=:.2f}")

    salario_anual_integrado = (salario_neto_mensual + 2 * fondo) * 12

    bono_anual, impuestos_bono = bono_anual_desemp(
        base_salary,
        p_bono_desemp=p_bono_desemp,
    )

    salario_anual_integrado += bono_anual - impuestos_bono

    startdate = datetime.date(year=2023, month=3, day=16)
    aguinaldo_date = datetime.date(year=2024, month=12, day=16)

    monto_aguinaldo = aguinaldo(
        base_salary, startdate, aguinaldo_date, dias_aguinaldo=dias_aguinaldo
    )
    monto_vacaciones = dias_vacaciones * prima_vacacional * base_salary / 30

    salario_anual_integrado += monto_aguinaldo + monto_vacaciones + extra_anuales

    print(
        f"{monto_aguinaldo=:.2f}, {monto_vacaciones=:.2f}, {bono_anual=:.2f}, {impuestos_bono=:.2f}, {extra_anuales=:.2f}"
    )

    salario_mensual_integrado = salario_anual_integrado / 12

    print(f"{salario_anual_integrado=:.2f}")
    print(f"{salario_mensual_integrado=:.2f}")


bluetab_prestaciones = dict(
    base_salary=40000,
    bono_puntualidad=0.1,
    bono_asistencia=0.1,
    monto_teletrabajo=133,
    dias_aguinaldo=15,
    prima_vacacional=0.25,
    dias_vacaciones=18,
    p_bono_desemp=0.08,
    monto_vales=2047.0,
    p_fondo=0.03,
)

scotiabank_prestaciones = dict(
    base_salary=37000,
    bono_puntualidad=0.0,
    bono_asistencia=0.0,
    monto_teletrabajo=0.0,
    dias_aguinaldo=30,
    prima_vacacional=0.5,
    dias_vacaciones=20,
    p_bono_desemp=0.1,
    monto_vales=3302.00,
    p_fondo=0.00,
    extra_anuales=18989,  # Su cosa rara del IMSS.
)

ILSP_prestaciones = dict(base_salary=44810, bono_puntualidad=0.1, bono_asistencia=0.1)


GE_prestaciones = dict(
    base_salary=28874.84,
    dias_aguinaldo=30,
    dias_vacaciones=12,
    prima_vacacional=0.6,
    p_vales=0.1,
    p_fondo=0.13,
)

calcular_salario_integrado(**GE_prestaciones)
calcular_salario_integrado(**ILSP_prestaciones)
calcular_salario_integrado(**bluetab_prestaciones)
calcular_salario_integrado(**scotiabank_prestaciones)

pana_javo = scotiabank_prestaciones.copy()
pana_javo["base_salary"] = 63000

Javo_reckit = dict(
    base_salary=50210,
    dias_vacaciones=12,
    bono_asistencia=0.1,
    bono_puntualidad=0.1,
    monto_vales=1200,
    p_bono_desemp=0.08,
)

calcular_salario_integrado(**Javo_reckit)

calcular_salario_integrado(**pana_javo)

calcular_salario_integrado(**pana_javo)


def calcular_salario_base_desde_neto(
    neto_objetivo=45000,
    base_salary=30000,
    prima_vacacional=0.25,
    dias_vacaciones=12,
    dias_aguinaldo=15,
    bono_puntualidad=0.0,
    bono_asistencia=0.0,
    monto_vales: float | None = None,
    p_vales: float = 0.0,
):
    salario_neto = 0
    while salario_neto < neto_objetivo:
        base_salary += 5

        if monto_vales is None:
            monto_vales, excedente = vales_despensa(
                base_salary, p=p_vales
            )  # vales_despensa(2900, p = 1)
        else:
            excedente = 0

        isr = ISR(base_salary + excedente)
        cuota_IMSS = IMSS(
            base_salary,
            prima_vacacional=prima_vacacional,
            dias_vacaciones=dias_vacaciones,
            dias_aguinaldo=dias_aguinaldo,
        )
        fondo = fondo_ahorro(base_salary, p=0)

        salario_neto = (
            base_salary
            - isr
            - cuota_IMSS
            + monto_vales
            + fondo
            + bono_puntualidad * base_salary
            + bono_asistencia * base_salary
        )

    return salario_neto


calcular_salario_base_desde_neto(
    50000,
    dias_vacaciones=16,
    bono_asistencia=0.1,
    bono_puntualidad=0.1,
    monto_vales=1500,
)

# %% Impuestos Marzo por bono


# Get the current date and time
now = datetime.datetime.now()

# Get the current month and year
current_month = now.month
current_year = now.year

print("Current month:", current_month)
print("Current year:", current_year)

startdate = datetime.date(year=2023, month=3, day=16)

aguinaldo_date = datetime.date(year=2023, month=12, day=16)

worked_days = np.busday_count(startdate, aguinaldo_date)

cuenta_fondo = 0
savings = 0
data = []
# Increase the month by one for the next 12 months
for i in range(1, 24):
    # Calculate the next month and year
    next_month = (current_month + i - 1) % 12 + 1
    next_year = current_year + ((current_month + i - 1) // 12)  # integer division

    # Create a new datetime object for the next month and year
    date = datetime.date(year=next_year, month=next_month, day=16)

    # Print the next date
    print("Month:", date.strftime("%B %Y"))

    base_salary = 28874.84
    ingresos_tributarios = base_salary
    ingresos_no_tributarios = 0
    isr = 0

    monto_vales, excedente = vales_despensa(
        base_salary, p=0.1
    )  # vales_despensa(2900, p = 1)
    ingresos_no_tributarios += monto_vales
    ingresos_tributarios += excedente

    fondo = fondo_ahorro(base_salary)
    ingresos_no_tributarios -= fondo
    cuenta_fondo += 2 * fondo

    if date.month == 3 and (date.year - 2023) >= 1:
        bono_desemp, impuestos = bono_anual_desemp(base_salary)
        ingresos_no_tributarios += bono_desemp
        isr += impuestos

    if date.month == 12:
        monto_aguinaldo = aguinaldo(base_salary, startdate, date, dias_aguinaldo=30)
        ingresos_tributarios += monto_aguinaldo

    if date.month == 12 or (date.month == 7):  # and date.year >= 2024):
        ingresos_no_tributarios += cuenta_fondo
        cuenta_fondo = 0

    isr += ISR(ingresos_tributarios)
    cuota_IMSS = IMSS(
        ingresos_tributarios,
        prima_vacacional=0.6,
        dias_vacaciones=12,
        dias_aguinaldo=30,
    )

    percepcion_mensual = (
        ingresos_tributarios - isr - cuota_IMSS + ingresos_no_tributarios
    )

    savings += percepcion_mensual

    data.append(
        [
            date,
            ingresos_tributarios,
            ingresos_no_tributarios,
            percepcion_mensual,
            isr,
            cuota_IMSS,
            monto_vales,
            fondo,
            cuenta_fondo,
            savings,
        ]
    )

df_income = pd.DataFrame(
    data,
    columns=[
        "Date",
        "Ingresos Tributarios",
        "Ingresos no Tributarios",
        "Percepciones mensual",
        "ISR",
        "IMSS",
        "Vales Despensa",
        "Fondo mes",
        "Cuenta de Ahorro Fondo",
        "Ahorro",
    ],
)


df_income.plot.bar(x="Date", y="Percepciones mensual")
df_income.plot.bar(x="Date", y="Ingresos Tributarios")
df_income.plot.bar(x="Date", y="Ingresos no Tributarios")
df_income.plot.bar(x="Date", y="ISR")
# #percepcion_anual = (ingreso_bruto + vales_despensa)*12 + bono_desemp
# salary_range = np.linspace(lim_inf[0],lim_inf[-2]*1.25,50*len(lim_inf))
# npISR = np.vectorize(ISR)

# brute_salary =  npISR(salary_range)
# plt.plot(salary_range,brute_salary/salary_range)

# limite_inferior = np.array(lim_inf[:-2])
# cuota_porcentual = np.array(cuota[:-2])/limite_inferior

# plt.scatter(limite_inferior, cuota_porcentual)

array = np.arange(16 * 40).reshape(40, 16)

mask_outliers = np.all(
    (df_B10_GOLD[columns] < 1.003) & (df_B10_GOLD[columns] < 0.997), axis=1
)
# %%
