import makepath

import numpy as np

from pyFEM.core import Structure
from docxtpl import DocxTemplate
import pprint

pp = pprint.PrettyPrinter(sort_dicts=False)

camion_CC14 = {
    'separacion_ejes': [4.3, 4.3],
    'peso_ejes': [40, 160, 160],
    'carga_carril': 10.8,
    'separacion_ruedas': 1.8,
    'separacion_borde': 0.6
}

# tabla 3.6.1.1.2-1 -- Factores de presencia múltiple, m
factor_presencia_multiple = {1: 1.2, 2: 1}


def superestructura():
    # losa: losa
    # carpetaAsfaltica: carpeta asfaltica
    # Nb: cantidad de vigas, -
    # vigas: vigas de la superestructura
    # S: separación entre vigas, m
    # S_overhang: voladizo, m
    # L: longitud de la superestructura, m

    L = 12
    superestructura = {
        'losa': losa(),
        'carpetaAsfaltica': carpeta_asfaltica(),
        'Nb': 3,
        'vigas': viga_i({'L': L}),
        'S': 1.69,
        'S_overhang': 1.56,
        'L': 12
    }

    model = create_model(superestructura)

    parametro_rigidez_longitudinal(superestructura)

    factor_distribucion_momentos_viga_interior(superestructura)
    factor_distribucion_cortante_viga_interior(superestructura)
    factor_distribucion_momento_viga_exterior(superestructura)
    factor_distribucion_diseno(superestructura)

    # avalúo de cargas
    avaluo_carga(superestructura)

    # momentos flectores
    # cargas permanentes
    momentos_flectores_cargas_permanentes(superestructura, model)
    # carpeta asfaltica
    momentos_flectores_carpeta_asfaltica(superestructura, model)
    momentos_flectores_bordillos_barandas(superestructura, model)
    # carga viva vehicular
    momentos_flectores_carga_viva_vehicular(superestructura, model)

    # combinaciones de carga
    combinaciones_carga(superestructura)

    return superestructura


def carpeta_asfaltica(params={}):
    # γ: peso específico del asfalto, kN/m3
    # e: espesor de la carpeta asfáltica, m

    params['γ'] = 22  # kN/m3
    params['e'] = 0.075

    return params

def losa(params={}):
    # γ: peso especifico, kN/m3
    # f'c : resistencia del concreto, MPa
    # E: módulo de elasticidad, MPa
    
    # ancho: ancho de la losa, m
    # ts: espesor de la losa, m
    # haunch: distancia entre la parte inferior de la losa y la parte superior del alma de las vigas, m

    params['γ'] = 24  # kN/m3
    params["f'c"] = 28
    params['E'] = 4800 * params["f'c"] ** 0.5 * 1000

    params['ancho'] = 6.5
    params['ts'] = 0.22
    params['haunch'] = 0.05

    params['baranda'] = barandas()

    return params

def barandas(params={}):
    # barandas del puente

    # bordillo

    

    params['bordillo'] = bordillo()
    params['peso'] = 3 / 2

    return params

def bordillo(params={}):
    #
    ancho = params['ancho'] = 0.25
    ancho_superior = params['anchoSuperior'] = 0.2
    altura = params['altura'] = 0.3
    peso_especifico = params['γ'] = 24

    params['peso'] = peso_especifico * (ancho + ancho_superior) / 2 * altura

    return params
    


def viga_i(params={}):
    # IPE 600
    # L: longitud de la viga, m
    
    # γ: peso unitario, kN/m3
    # fy: resistencia del acero, MPa
    # E: módulo de elasticidad, kPa

    # tf: espesor de la aleta, m
    # bf: ancho de la aleta, m
    # tw: espesor del alma, m

    # H: altura total de la viga, m
    # D: altura libre del alma, m

    # ys: centroíde medido desde la parte superior de la viga, m
    # A: área, m2
    # I: inercia, m4

    # Dc: profundidad del alma en compresión en el rango elástico, m
    # Iyc: momento de inercia de la aleta en compresión respecto al eje del alma, m4
    # Iyt: momento de inercia de la aleta en tensión respecto al eje del alma, m4

    # peso: peso de la sección transversal de la viga, kN/m

    # λrw: relación de esbeltez límite para un alma no compacta, -

    L = params['L']

    peso_unitario = params['γ'] = 78.5  # kN/m3   
    fy = params['fy'] = 420
    E = params['E'] = 200000000

    tf = params['tf'] = 0.019
    bf = params['bf'] = 0.22
    tw = params['tw'] = 0.012

    H = params['H'] = 0.6
    D = params['D'] = 0.514

    ys = params['ys'] = 0.3
    area = params['A'] = 0.015600
    params['I'] = 0.000921

    Dc = params['Dc'] = ys - tf
    Iyc = params['Iyc'] = tf * bf ** 4 / 12
    Iyt = params['Iyt'] = tf * bf ** 4 / 12

    params['peso'] = peso_unitario * area


    λrw = params['λrw'] = 5.7 * (E / fy) ** 0.5

    # check
    check = {}

    check['0.033L<H'] = 0.033 * L < H
    check['bf/2tf<=12'] = bf / (2*tf) < 12
    check['bf>=D/6'] = bf >= D / 6
    check['tf>=1.1tw'] = tf >= 1.1*tw
    check['0.1<=Iyc/Iyt<=10'] = 0.1 <= Iyc / Iyt and Iyc / Iyt <= 10
    check['D/tw<150'] = D / tw < 150
    check['bfc>=L/85'] = bf >= L / 85
    check['2Dc/tw<=λrw'] = 2*Dc / tw <= λrw

    params['check'] = check

    return params


def parametro_rigidez_longitudinal(superestructura):
    # longitudinal stifness parameter
    losa = superestructura['losa']
    viga = superestructura['vigas']

    # calcular el parámetro de rigidez longitudinal
    Kg = {}

    n = Kg['n'] = viga['E'] / losa['E']
    I = Kg['I'] = viga['I']
    A = Kg['A'] = viga['A']
    eg = Kg['eg'] = losa['ts'] / 2 + (losa['haunch'] - viga['tf']) + viga['ys']

    Kg['Kg'] = n * (I + A * eg ** 2)

    superestructura['parametroRigidezLongitudinal'] = Kg

    return Kg


def factor_distribucion_momentos_viga_interior(superestructura):
    # g_int_moment_1: distribución de la carga viva por linea para momento en vigas interiores, -

    factor_distribucion = {}
    losa = superestructura['losa']

    Nb = superestructura['Nb']
    S = superestructura['S']
    L = superestructura['L']
    Kg = superestructura['parametroRigidezLongitudinal']['Kg']
    ts = losa['ts']

    # verificar la aplicabilidad de la ecuación
    check = {}

    check['1.1<=S<=4.9'] = 1.1 < S < 4.9
    check['0.11<=ts<=0.3'] = 0.11 < ts < 0.3
    check['6<=L<=73'] = 6 < L < 73
    check['Nb==3'] = Nb == 3
    check['Nb>=4'] = Nb >= 4
    check['0.0041623<=Kg<=2.9136'] = 0.0041623 <= Kg <= 2.9136

    factor_distribucion['check'] = check

    if (check['1.1<=S<=4.9'] and
        check['0.11<=ts<=0.3'] and
        check['6<=L<=73'] and
        check['0.0041623<=Kg<=2.9136'] and
        check['Nb==3'] or check['Nb>=4']):

        eq = {}

        g_int_moment_1 = eq['g_int_moment_1'] = 0.06 + (S / 4.3)**0.4 * (S / L)**0.3 * (Kg / (L * ts ** 3))**0.1
        g_int_moment_2 = eq['g_int_moment_2'] = 0.075 + (S / 2.9)**0.6 * (S / L)**0.2 * (Kg / (L * ts ** 3))**0.1
        eq['g_int_moment'] = max(g_int_moment_1, g_int_moment_2)

        factor_distribucion['ecuacion'] = eq

    if check['Nb==3']:
        regla_palanca = {}

        separacion_ruedas = camion_CC14['separacion_ruedas']
        separacion_borde = camion_CC14['separacion_borde']
        
        # un carril cargado
        m = factor_presencia_multiple[1]
        g_int_moment_1 = regla_palanca['g_int_moment_1'] = m * (2*S - separacion_ruedas) / (2*S)

        # dos carriles cargados
        m = factor_presencia_multiple[2]
        g_int_moment_2 = regla_palanca['g_int_moment_2'] = 2 * m * (2*S - 2*separacion_borde - separacion_ruedas) / (2*S)

        regla_palanca['g_int_moment'] = max(g_int_moment_1, g_int_moment_2)

        factor_distribucion['reglaPalanca'] = regla_palanca

    factor_distribucion['g_int_moment'] = min(factor_distribucion['ecuacion']['g_int_moment'], factor_distribucion['reglaPalanca']['g_int_moment'])

    superestructura['factorDistribucion'] = factor_distribucion

def factor_distribucion_cortante_viga_interior(superestructura):
    factor_distribucion = {}

    Nb = superestructura['Nb']
    S = superestructura['S']

    # verificar la aplicabilidad de la ecuación
    check = {}
    check['Nb==3'] = Nb == 3
    factor_distribucion['check'] = check

    if check['Nb==3']:
        separacion_ruedas = camion_CC14['separacion_ruedas']
        separacion_borde = camion_CC14['separacion_borde']
        
        # un carril cargado
        m = factor_presencia_multiple[1]
        g_int_moment_1 = factor_distribucion['g_int_1'] = m * (2*S - separacion_ruedas) / (2*S)

        # dos carriles cargados
        m = factor_presencia_multiple[2]
        g_int_moment_2 = factor_distribucion['g_int_2'] = 2 * m * (2*S - 2*separacion_borde - separacion_ruedas) / (2*S)

        factor_distribucion['g_int'] = max(g_int_moment_1, g_int_moment_2)

    superestructura['factorDistribucionCortanteVigaInterior'] = factor_distribucion

def factor_distribucion_momento_viga_exterior(superestructura):
    # factor de distrubción de momento flector en la viga exterior
    factor_distribucion = {}

    Nb = superestructura['Nb']
    S = superestructura['S']
    S_overhang = superestructura['S_overhang']
    ancho_bordillo = superestructura['losa']['baranda']['bordillo']['ancho']
    d_e = factor_distribucion['d_e'] = S_overhang - ancho_bordillo

    # verificar la aplicabilidad de la ecuación
    check = {}

    check['Nb==3'] = Nb == 3
    check['-0.3<=de<=1.7'] = -0.3 <= d_e <= 1.7

    factor_distribucion['check'] = check

    separacion_ruedas = camion_CC14['separacion_ruedas']
    separacion_borde = camion_CC14['separacion_borde']

    if check['-0.3<=de<=1.7']:
        eq = {}

        # un carril cargado
        m = factor_presencia_multiple[1]
        mg_1 = eq['mg_1'] = m * (2*(ancho_bordillo + separacion_borde - S_overhang) + separacion_ruedas) / (2*S)

        # dos carriles cargados
        mg_int = superestructura['factorDistribucion']['g_int_moment']
        e = 0.77 + d_e/ 2.8
        mg_2 = eq['mg_2'] = e * mg_int 

        factor_distribucion['e'] = e
        factor_distribucion['mg_int'] = mg_int
        eq['mg'] = max(mg_1, mg_2)


        factor_distribucion['ecuacion'] = eq

    if check['Nb==3']:
        # regla de la palanca
        regla_palanca = {}

        # un carril cargado
        mg_ext_momento_1 = regla_palanca['mg_Me_1C'] = mg_1

        # dos carriles cargados
        m = factor_presencia_multiple[2]
        mg_ext_momento_2 = regla_palanca['mg_Me_2C'] = 2 * m * (3 * (S_overhang + S) - 4*separacion_borde - 3*ancho_bordillo - 2*separacion_ruedas) / (2*S)

        regla_palanca['mg_Me'] = max(mg_ext_momento_1, mg_ext_momento_2)

        factor_distribucion['reglaPalanca'] = regla_palanca

    factor_distribucion['mg_Me'] = min(factor_distribucion['ecuacion']['mg'], factor_distribucion['reglaPalanca']['mg_Me'])

    superestructura['factorDistribucionMomentoVigasExteriores'] = factor_distribucion

def factor_distribucion_diseno(superestructura):
    factor_distribucion_viga_interior = superestructura['factorDistribucion']['g_int_moment']
    factor_distribucion_viga_exterior = superestructura['factorDistribucionMomentoVigasExteriores']['mg_Me']

    superestructura['factorDistribucionDiseno'] = max(factor_distribucion_viga_interior, factor_distribucion_viga_exterior)

def avaluo_carga(superestructura):
    #
    avaluo_carga = {}

    # carga muerta
    carga_muerta = avaluo_carga['cargaMuerta'] = {}
    # losa
    espesor_losa = superestructura['losa']['ts']
    peso_especifico = superestructura['losa']['γ']
    ancho_aferente = superestructura['S']
    
    losa = carga_muerta['losa'] = peso_especifico * espesor_losa * ancho_aferente
    # viga
    viga = carga_muerta['viga'] = superestructura['vigas']['peso']

    # personal y equipos
    personalEquipos = carga_muerta['personalEquipos'] = 0.75 # kN/m

    # total
    carga_muerta['total'] = losa + viga + personalEquipos

    # sobreimpuesta
    carga_sobreimpuesta = avaluo_carga['cargaSobreimpuesta'] = {}
    # carpeta asfaltica
    no_vigas = superestructura['Nb']
    peso_especifico = superestructura['carpetaAsfaltica']['γ']
    espesor = superestructura['carpetaAsfaltica']['e']
    ancho_losa = superestructura['losa']['ancho']
    ancho_bordillo = superestructura['losa']['baranda']['bordillo']['ancho']

    carpetaAsfaltica = carga_sobreimpuesta['carpetaAsfaltica'] = peso_especifico * espesor * (ancho_losa - 2 * ancho_bordillo) / no_vigas
    # bordillo + baranda
    peso_bordillo = superestructura['losa']['baranda']['bordillo']['peso']
    peso_baranda = superestructura['losa']['baranda']['peso']

    bordilloBaranda = carga_sobreimpuesta['bordilloBaranda'] = 2 * (peso_bordillo + peso_baranda) / no_vigas
    # 


    # total
    # carga_sobreimpuesta['total'] = carpetaAsfaltica + bordilloBaranda

    superestructura['avaluoCarga'] = avaluo_carga

    return avaluo_carga


def create_model(superestructura):
    # pyFEM model
    model = Structure(uy=True, rz=True)

    # add material
    model.add_material(1, E=superestructura['vigas']['E'])

    # add section
    model.add_section(1, Iz=superestructura['vigas']['I'])

    # add joints
    model.add_joint(1, x=0)
    model.add_joint(2, x=superestructura['L'])

    # add frame
    model.add_frame(1, 1, 2, 1, 1)

    # add supports
    model.add_support(1, uy=True)
    model.add_support(2, uy=True)

    # add load patterns
    model.add_load_pattern('test')
    model.add_distributed_load('test', frame=1, fy=-1)

    return model

def momentos_flectores_cargas_permanentes(superestructura, model):
    carga_permanente = superestructura['avaluoCarga']['cargaMuerta']['total']

    frame = model.frames[1]
    length = frame.get_length()
    loadPattern = model.add_load_pattern("cargaPermanente")
    loadPattern.add_distributed_load(frame, fy=-carga_permanente)

    model.solve()

    mz = model.internal_forces[loadPattern][frame].mz
    n = len(mz)

    superestructura['momentosFlectoresCargasPermanentes'] = [[i / (n - 1) * length, m] for i, m in enumerate(mz)]

def momentos_flectores_carpeta_asfaltica(superestructura, model):
    carga_carpeta = superestructura['avaluoCarga']['cargaSobreimpuesta']['carpetaAsfaltica']

    frame = model.frames[1]
    length = frame.get_length()

    loadPattern = model.add_load_pattern('carpetaAsfaltica')
    loadPattern.add_distributed_load(frame, fy=-carga_carpeta)

    model.solve()

    mz = model.internal_forces[loadPattern][frame].mz
    n = len(mz)

    superestructura['momentosFlectoresCarpetaAsfaltica'] = [[i / (n - 1) * length, m] for i, m in enumerate(mz)]


def momentos_flectores_bordillos_barandas(superestructura, model):
    carga_bordillosBarandas = superestructura['avaluoCarga']['cargaSobreimpuesta']['bordilloBaranda']

    frame = model.frames[1]
    length = frame.get_length()

    loadPattern = model.add_load_pattern('cargaBordillosBarandas')
    loadPattern.add_distributed_load(frame, fy=-carga_bordillosBarandas)

    model.solve()

    mz = model.internal_forces[loadPattern][frame].mz
    n = len(mz)

    superestructura['momentosFlectoresBordillosBarandas'] = [[i / (n - 1) * length, m] for i, m in enumerate(mz)]


def momentos_flectores_carga_viva_vehicular(superestructura, model):
    frames = {frame: key for key, frame in model.frames.items()}

    frame = model.frames[1]
    length = frame.get_length()

    x_ejes = np.array([0] + camion_CC14['separacion_ejes'])

    for i in range(x_ejes.shape[0]):
        for j in range(i+1, len(x_ejes)):
            x_ejes[j] += x_ejes[i]
    
    length_camion = x_ejes[-1]
    casos_carga_camion_CC14 = []
    n = 21 # NUMERO PARADAS CAMION
    peso_ejes = camion_CC14['peso_ejes']
    for i in range(n):
        x = (i / (n - 1)) * (length + length_camion)

        loadPattern = model.add_load_pattern('{:.3f} m'.format(x))

        for i, xi in enumerate(x - x_ejes):
            if length > xi > 0:
                loadPattern.add_point_load_at_frame(frame, fy=(-peso_ejes[i], xi / length))

        casos_carga_camion_CC14.append(model.load_patterns['{:.3f} m'.format(x)])

    # carga carril
    w = -camion_CC14['carga_carril']
    loadPattern = model.add_load_pattern('carril')
    loadPattern.add_distributed_load(frame, fy=w)

    model.solve()

    n = 11  # cantidad de valores por elemento por defecto
    momentos = {(i / (n - 1) * length): [] for i in range(n)}
    for load_pattern in casos_carga_camion_CC14:
        mz = model.internal_forces[load_pattern][frame].mz
        n = len(mz)

        for i in range(n):
            x = (i / (n - 1)) * length
            momentos[x].append(mz[i])

    momentos_maximos = []
    for x, m in momentos.items():
        momentos_maximos.append([x, max(m)])

    loadPattern = model.load_patterns['carril']
    momentos_carril = []
    for i, m in enumerate(model.internal_forces[loadPattern][frame].mz):
        momentos_carril.append([(i / (n - 1)) * length, m])


    momentos_carga_vehicular = []
    factor_distribucion = superestructura['factorDistribucionDiseno']
    for i, (camion, carril) in enumerate(zip(momentos_maximos, momentos_carril)):
        x = camion[0]
        momentos_carga_vehicular.append([x, factor_distribucion * (1.33 * camion[1] + carril[1])])

    superestructura['momentosFlectoresCamion'] = momentos_maximos
    superestructura['momentosFlectoresCarril'] = momentos_carril
    superestructura['momentosFlectoresCargaVehicular'] = momentos_carga_vehicular

def combinaciones_carga(superestructura):
    superestructura['combinacionesCarga'] = {}

    length = superestructura['L']
    n = 11

    m_carga_permanente = superestructura['momentosFlectoresCargasPermanentes']
    m_bordillos_barandas = superestructura['momentosFlectoresBordillosBarandas']
    m_carpeta_asfaltica = superestructura['momentosFlectoresCarpetaAsfaltica']
    m_carga_vehicular = superestructura['momentosFlectoresCargaVehicular']
    cargas = [m_carga_permanente, m_bordillos_barandas, m_carpeta_asfaltica, m_carga_vehicular]

    # estado límite de resistencia última
    mu = []
    for i, (carga_permanente, bordillos_barandas, carpeta_asfaltica, carga_vehicular) in enumerate(zip(*cargas)):
        mu.append([(i / (n - 1) * length), 1.25 * (carga_permanente[1] + bordillos_barandas[1]) + 1.5 * carpeta_asfaltica[1] +1.75 * carga_vehicular[1]])

    superestructura['combinacionesCarga']['resistenciaUltima'] = mu

    # estado límite de resistencia IV
    mu = []
    for x, m in m_carga_permanente:
        mu.append([x, 1.5 * m])

    superestructura['combinacionesCarga']['resistenciaIV'] = mu

    # estado límite servicio
    mu = []
    for i, (carga_permanente, bordillos_barandas, carpeta_asfaltica, carga_vehicular) in enumerate(zip(*cargas)):
        mu.append([(i / (n - 1) * length), 1 * (carga_permanente[1] + bordillos_barandas[1]) + 1 * carpeta_asfaltica[1] +1 * carga_vehicular[1]])

    superestructura['combinacionesCarga']['servicio'] = mu
    

if __name__ == '__main__':
    superestructura = superestructura()
    # pp.pprint(superestructura)

    # doc template
    doc = DocxTemplate('Memoria de cálculos de las vigas metálicas del puente.docx')

    doc.render(superestructura)
    doc.save('output.docx')
