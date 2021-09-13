import makepath

from pyFEM import Structure
from docxtpl import DocxTemplate

import numpy as np
import matplotlib.pyplot as plt
import pprint

pp = pprint.PrettyPrinter(sort_dicts=False)

camion_CC14 = {
    'separacion_ejes': [4.3, 4.3],
    'peso_ejes': [40, 160, 160],
    'carga_carril': 10.3,
    'separacion_ruedas': 1.8,
    'separacion_borde': 0.6
}

# tabla 3.6.1.1.2-1 -- Factores de presencia múltiple, m
factor_presencia_multiple = {1: 1.2, 2: 1}


def superestructura(params={}):
    # losa: losa
    # carpetaAsfaltica: carpeta asfaltica
    # Nb: cantidad de vigas, -
    # vigas: vigas de la superestructura
    # S: separación entre vigas, m
    # S_overhang: voladizo, m
    # L: longitud de la superestructura, m

    # svigas = 2
    # bf = 'bf'
    # elosa = 
    # pesoconcreto = 

    params = {}
    nb = params['nb'] = params.get('nb', 4) # numero de vigas 
    nl = params['nl'] = params.get('nl', 2) # numero de carriles cargados
    fy = params['fy'] = params.get('fy', 420000) # acero en kPa
    fc = params['fc'] = params.get('fc', 28000)  # concreto en kPa
    tiposeccion = params['tiposeccion'] = params.get('tiposeccion', 'e') # tipo de seccion del puente
    factormodcarga = params['factormodcarga'] = params.get('factormodcarga', 1) # factor de modificacion de carga
    L = params['L'] = params.get('L', 14) # Longitud de la luz
    svigas = params['svigas'] = params.get('svigas', 2) # Separacion entre vigas
    distvoladizo = params['distvoladizo'] = params.get('distvoladizo', 1) # distancia del voladizo desde el centro de la viga
    baseviga = params['baseviga'] = params.get('baseviga', 0.4) # base de la viga
    hviga = params['hviga'] = params.get('hviga', 0.8) # altura de la viga
    elosa = params['elosa'] = params.get('elosa', 0.2) # espesor de la losa
    pesoconcreto = params['pesoconcreto'] = params.get('pesoconcreto', 23.54) # peso especifico del concreto
    pesoasfalto = params['pesoasfalto'] = params.get('pesoasfalto', 21.57) # peso especifico del asfalto
    pesobaranda = params['pesobaranda'] = params.get('pesobaranda', 0.6865) # peso especifico de la baranda
    nbarandas = params['nbaranda'] = params.get('nbaranda', 2) # numero de barandas
    nbordillo = params['nbordillo'] = params.get('nbordillo', 2) # numero de bordillos
    seccionbordillo1 = params['seccionbordillo1'] = params.get('seccionbordillo1', 0.2) # ancho bordillo
    seccionbordillo2 = params['seccionbordillo2'] = params.get('seccionbordillo2', 0.3) # alto bordillo
    ecarpetaasf = params['ecarpetaasf'] = params.get('ecarpetaasf', 0.08) # espesor carperta asfaltica
    MLv = params['MLv'] = params.get('MLv', 843) # valor asumido del momento generado por el vehiculo, debe ser calculado
    MLc = params['MLc'] = params.get('MLc', 252) # valor asumido del momento generado por el carril, debe ser calculado
    IM = params['IM'] = params.get('IM', 1.33)  # factor de amplificacion dinamica de carga
    n = params['n'] = params.get('n', 1) # Relacion modular
    b1 = params['b1'] = params.get('b1', 2.2) # Distancia para el calculo del factor de distribucion regla de la palanca
    b2 = params['b2'] = params.get('b2', 0.4) # Distancia para el calculo del factor de distribucion regla de la palanca
    rec = params['rec'] = params.get('rec', 0.1) # recubrimiento del acero
    frf = params['frf'] = params.get('frf', 0.9) # factor de resistencia por flexion
    rbarra = params['rbarra'] = params.get('rbarra', 8) # referencia de la barra
    abarra = params['abarra'] = params.get('abarra', 0.000510) # area de la barra en metros
    pb1 = params['pb1'] = params.get('pb1', 0.85) # valor para el calculo dela profundidad del bloque de compresiones
    duc = params['duc'] = params.get('duc', 0.003) # deformacion unitaria del concreto
    duas = params['duas'] = params.get('duas', 0.005) # deformacion unitaria del acero supuesta
    y3 = params['y3'] = params.get('y3', 0.75) # valor del concreto para el momento requerido por la combinacion de carga
    y1 = params['y1'] = params.get('y1', 1.6) # valor del concreto para el momento requerido por la combinacion de carga

    Ec = params['Ec'] = 4800 * (fc / 1000) ** 0.5 #Modulo de elasticidad del concreto
    hmin = params['hmin'] = 0.07*L # altura minima
    hseccion = params['hseccion'] = hviga + elosa  # altura de la seccion compuesta
    bf = params['bf'] = (svigas/2) + distvoladizo # ancho efectico de la aleta
    DClosa = params['DClosa'] = bf*elosa*pesoconcreto # carga de la losa
    DCviga = params['DCviga'] = baseviga*hviga*pesoconcreto # carga de la viga
    DCest = params['DCest'] = DClosa + DCviga # carga de la estructura
    DCbaranda = params['DCbaranda'] = (pesobaranda*nbarandas)/nb # carga de la baranda
    DCbordillo = params['DCbordillo'] = (seccionbordillo1*seccionbordillo2*pesoconcreto*nbordillo)/nb # carga del bordillo
    DCper = params['DCper'] = DClosa + DCviga + DCbordillo + DCbaranda # carga DC
    DW = params['DW'] = ecarpetaasf*bf*pesoasfalto # carga del asfalto   
    MDCest = params['MDCest'] = (DCest*(14**2))/8 # Momento maximo estructura
    MDW = params['MDW'] = (DW*(14**2))/8 # Momento maximo del asfalto
    MDCvol = params['MDCvol'] = ((DCbordillo + DCbaranda)*(14**2))/8 # Momento maximo del voladizo
    MDCper = params['MDCper'] = MDCest + MDCvol # Momento maximo carga DC
    MLLIM = params['MLLIM'] = (IM*MLv) + MLc # Momento maximo carga viva vehicular
    A = params['A'] = baseviga*hviga # Area de la viga seccion simple
    y = params['y'] = hviga/2 # centroide la viga seccion simple
    Al = params['Al'] = bf*elosa # Area de la losa seccion simple
    yl = params['yl'] = elosa/2 # centroide la losa seccion simple
    Ac = params['Ac'] = A + Al # Area de la seccion compuesta
    yc = params['yc'] = ((A*y)+(Al*(yl+hviga)))/Ac # centroide la seccion compuesta
    I = params['I'] = (baseviga*(hviga**3))/12 # inercia de la viga sección simple
    Il = params['Il'] = (bf*(elosa**3))/12 # inercia de la losa sección simple
    Ic = params['Ic'] = (I +(A*((yc-y)**2))) + (Il +(Al*(((yl+hviga)-yc)**2)))  # inercia de la sección compuesta
    Snc = params['Snc'] = I/y # modulo de la seccion simple
    Sc = params['Sc'] = Ic/yc # modulo de la seccion compuesta
    eg = params['eg'] = hseccion-(elosa/2)-(hviga/2) # distancia entre centroides de la viga y la losa
    kg = params['kg'] = n*(I + (A*(eg**2))) # Parametro para el calculo del factor de distribucion
    de = params['de'] = distvoladizo - seccionbordillo1 # distancia entre eje de la viga exterior y la cara interna de la bordillo
    mg1i = params['mg1i'] = 0.06 + ((svigas/4.3)**0.4)*((svigas/L)**0.3)*((kg/(L*elosa**3))**0.1) # factor de distribucion
    mg2i = params['mg2i'] = 0.075 + ((svigas/2.9)**0.6)*((svigas/L)**0.2)*((kg/(L*elosa**3))**0.1) # factor de distribucion
    g1e = params['g1e'] =  (b1 + b2)/(2*svigas) # factor de distribucion sin mayorar por el factor de presencia multiple
    mg1e = params['mg1e'] = 1.2*0.65 # factor de distribucion
    mg2e = params['mg2e'] = (0.77 + (de/2.80))*mg2i # factor de distribucion
    MLLIMp = params['MLLIMp'] = MLLIM*mg2e # momento maximo debido a la carga viva con el factor de distribucion maximo hallado
    MUI = params['MUI'] = factormodcarga*((1.25*MDCper)+(1.5*MDW)+(1.75*MLLIMp)) # momento ultimo para resistencia I
    d = params['d'] = hseccion - rec # altura efectiva
    k = params['k'] = MUI/(bf*(d**2)) # parametro K para la cuantia
    m = params['m'] = fy/(0.85*fc) # parametro m para la cuantia
    p = params['p'] = (1/m)*(1-(1-((2*m*k)/(frf*fy)))**0.5) # cuantia
    As = params['As'] = p*d*bf # acero de refuerzo
    nbarra = params['nbarra'] = As/abarra # numero de barras a usar
    a = params['a'] = (p*d*fy)/(0.85*fc) # posicion del eje neutro
    pc = params['pc'] = (As*fy)/(0.85*fc*bf*pb1) # profundidad del bloque de compresiones
    dua = params['dua'] = (d-pc)*(duc/pc) # deformacion unitaria del acero

    # 'losa': losa(),
    # 'carpetaAsfaltica': carpeta_asfaltica(),
    # 'Nb': 3,
    # 'vigas': viga_i({'L': L}),
    # 'S': 1.69,
    # 'S_overhang': 1.56,
    # 'L': L,
    # 'baseviga': 0.4,
    # 'fy': 420000,
    # 'fc': 28000,
    # 'tipodeseccion': 'e',
    # 'factormodcarga': 1,
    # 'svigas': svigas,
    # 'bf': (svigas / 2 ) * 2,
    # 'hviga': 0.8,
    # 'elosa': 0.2,
    # 'pesoconcreto': 2.4,
    
    # 'DCviga': 0.2,
    # 'DCbordillo': 0.2,
    # 'DCbaranda': 0.2,
    # 'DW': 0.2,
    # }

    # superestructura['']
    # 
    model = create_model(params)

    # mz = model.internal_forces['DC'][1].mz
    # x = np.linspace(0, L, len(mz))
    # params['MDC'] = [[x, m] for x, m in zip(x, mz)]
    # mz = model.internal_forces['DW'][1].mz
    # x = np.linspace(0, L, len(mz))
    # params['MDW'] = [[x, m] for x, m in zip(x, mz)]

    # parametro_rigidez_longitudinal(superestructura)

    # factor_distribucion_momentos_viga_interior(superestructura)
    # factor_distribucion_cortante_viga_interior(superestructura)
    # factor_distribucion_momento_viga_exterior(superestructura)
    # factor_distribucion_diseno(superestructura)

    # avalúo de cargas
    # avaluo_carga(superestructura)

    # momentos flectores
    # cargas permanentes
    momentos_flectores_cargas_estructura(params, model)
    momentos_flectores_cargas_permanentes(params, model)
    # carpeta asfaltica
    # momentos_flectores_carpeta_asfaltica(superestructura, model)
    # momentos_flectores_bordillos_barandas(superestructura, model)
    # carga viva vehicular
    momentos_flectores_carga_viva_vehicular(params, model)

    # combinaciones de carga
    # combinaciones_carga(superestructura)

    return params


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
    L = superestructura['L']
    Ec = superestructura['Ec']
    Iz = superestructura['I']
    DCper = superestructura['DCper']
    DW = superestructura['DW']

    model = Structure(uy=True, rz=True)

    # add material
    model.add_material(1, E=Ec)

    # add section
    model.add_section(1, Iz=Iz)

    # add joints
    model.add_joint(1, x=0)
    model.add_joint(2, x=L)

    # add frame
    model.add_frame(1, 1, 2, 1, 1)

    # add supports
    model.add_support(1, uy=True)
    model.add_support(2, uy=True)

    model.set_flags_active_joint_displacements()
    model.set_indexes()
    model.set_stiffness_matrix_modified_by_supports()

    # add load patterns
    # model.add_load_pattern('DC')
    # model.add_distributed_load('DC', 1, fy=-DCper)
    
    # model.add_load_pattern('DW')
    # model.add_distributed_load('DW', 1, fy=-DW)
    
    # model.solve()

    return model

def plot(x, y, name, title, ylabel):
    # loadPattern = model.load_patterns['MDC']
    # mz = model.internal_forces[loadPattern][frame].mz

    fig, ax = plt.subplots()

    ax.plot(x, y, 'r')
    ax.set_title(title)  # 'Momento flector'
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(ymax=0)

    ax.set_xlabel('m')
    ax.set_ylabel(ylabel)  # 'kN m'

    ax.set_xticks(x[::2])
    ax.set_yticks(list(set(np.round_(y[::2], 3))))
    ax.grid(True)

    fig.savefig(f'{name}.png')

def momentos_flectores_cargas_estructura(params, model):
    carga_estructura = params['DCest']  # superestructura['avaluoCarga']['cargaMuerta']['total']

    frame = model.frames[1]
    length = frame.get_length()
    loadPattern = model.add_load_pattern("cargaPermanenteest")
    loadPattern.add_distributed_load(frame.name, fy=-carga_estructura)

    model.solve_load_pattern(loadPattern.name)

    mz = frame.get_internal_forces(loadPattern.name, 20)['mz'] # model.internal_forces[loadPattern.name][frame.name].mz
    x = np.linspace(0, length, len(mz))


    plot(x, -mz, 'Mdnc', 'Momentos flectores', 'kN m')

    x = x[::2]
    mz = mz[::2]
    params['Mdnc'] = [[x, m] for x, m in zip(x, mz)]

def momentos_flectores_cargas_permanentes(params, model):
    carga_permanente = params['DCper']  # superestructura['avaluoCarga']['cargaMuerta']['total']

    frame = model.frames[1]
    length = frame.get_length()
    loadPattern = model.add_load_pattern("cargaPermanente")
    loadPattern.add_distributed_load(frame.name, fy=-carga_permanente)

    model.solve_load_pattern(loadPattern.name)

    mz = frame.get_internal_forces(loadPattern.name, 20)['mz'] # model.internal_forces[loadPattern.name][frame.name].mz
    x = np.linspace(0, length, len(mz))


    plot(x, -mz, 'MDC', 'Momentos flectores', 'kN m')

    x = x[::2]
    mz = mz[::2]
    params['MDC'] = [[x, m] for x, m in zip(x, mz)]




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


def momentos_flectores_carga_viva_vehicular(params, model):
    frame = model.frames[1]
    length = frame.get_length()

    x_ejes = np.array([0] + camion_CC14['separacion_ejes'])

    for i in range(x_ejes.shape[0]):
        for j in range(i+1, len(x_ejes)):
            x_ejes[j] += x_ejes[i]
    
    length_camion = x_ejes[-1]
    casos_carga_camion_CC14 = []
    n = 41 # NUMERO PARADAS CAMION
    peso_ejes = camion_CC14['peso_ejes']
    for i in range(n):
        x = (i / (n - 1)) * (length + length_camion)

        loadPattern = model.add_load_pattern('{:.3f} m'.format(x))

        for i, xi in enumerate(x - x_ejes):
            if length > xi > 0:
                loadPattern.add_point_load_at_frame(frame.name, fy=(-peso_ejes[i], xi / length))

        casos_carga_camion_CC14.append(loadPattern.name) # model.load_patterns['{:.3f} m'.format(x)]

    # carga carril
    w = -camion_CC14['carga_carril']
    loadPattern = model.add_load_pattern('carril')
    loadPattern.add_distributed_load(frame.name, fy=w)

    model.solve()

    n = 11  # cantidad de valores por elemento por defecto
    momentos = {(i / (n - 1) * length): [] for i in range(n)}
    for load_pattern in casos_carga_camion_CC14:
        mz = model.internal_forces[load_pattern][frame.name].mz
        n = len(mz)

        for i in range(n):
            x = (i / (n - 1)) * length
            momentos[x].append(mz[i])

    momentos_maximos = []
    for x, m in momentos.items():
        momentos_maximos.append([x, max(m)])

    loadPattern = model.load_patterns['carril']
    momentos_carril = [[(i / (n - 1)) * length, m] for i, m in enumerate(model.internal_forces[loadPattern.name][frame.name].mz)]

    momentos_carga_vehicular = []
    # factor_distribucion = params['factorDistribucionDiseno']
    for i, (camion, carril) in enumerate(zip(momentos_maximos, momentos_carril)):
        x = camion[0]
        momentos_carga_vehicular.append([x, 1.33 * camion[1] + carril[1]])  # factor_distribucion * ()
    
    params['MLV'] = momentos_maximos
    params['MLC'] = momentos_carril
    params['MLL'] = momentos_carga_vehicular

    params['MLVmax'] = max([m[1] for m in momentos_maximos])
    params['MLCmax'] = max([m[1] for m in momentos_carril])
    params['MLLmax'] = max([m[1] for m in momentos_carga_vehicular])

    return params
    

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
    doc = DocxTemplate('template.docx')
    doc.render(superestructura)
    doc.save('output.docx')
    print('todo ok')
