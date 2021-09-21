from numpy.lib.shape_base import vsplit
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

tandem_CC14 = {
    'separacion_ejes': [1.2],
    'peso_ejes': [125, 125],
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

    nb = params['nb'] = params.get('nb', 4) # numero de vigas 
    nl = params['nl'] = params.get('nl', 2) # numero de carriles cargados
    fy = params['fy'] = params.get('fy', 420000) # acero en kPa
    fc = params['fc'] = params.get('fc', 28000)  # concreto en kPa
    Es = params['Es'] = params.get('Es', 200000000)  # modulo de elasticidad del acero en kPa 
    tiposeccion = params['tiposeccion'] = params.get('tiposeccion', 'e') # tipo de seccion del puente
    frf = params['frf'] = params.get('frf', 0.9) # factor de resistencia por flexion y cortante
    factormodcarga = params['factormodcarga'] = params.get('factormodcarga', 1) # factor de modificacion de carga
    L = params['L'] = params.get('L', 14) # Longitud de la luz
    aapoyo = params['aapoyo'] = params.get('aapoyo', 0.4) # ancho del apoyo
    svigas = params['svigas'] = params.get('svigas', 2) # Separacion entre vigas
    distvoladizo = params['distvoladizo'] = params.get('distvoladizo', 1) # distancia del voladizo desde el centro de la viga
    baseviga = params['baseviga'] = params.get('baseviga', 0.4) # base de la viga
    hviga = params['hviga'] = params.get('hviga', 0.8) # altura de la viga
    elosa = params['elosa'] = params.get('elosa', 0.2) # espesor de la losa
    pesoconcreto = params['pesoconcreto'] = params.get('pesoconcreto', 24) # peso especifico del concreto
    pesoasfalto = params['pesoasfalto'] = params.get('pesoasfalto', 21.57) # peso especifico del asfalto
    pesobaranda = params['pesobaranda'] = params.get('pesobaranda', 0.6865) # peso especifico de la baranda
    nbarandas = params['nbarandas'] = params.get('nbarandas', 2) # numero de barandas
    nbordillo = params['nbordillo'] = params.get('nbordillo', 2) # numero de bordillos
    seccionbordillo1 = params['seccionbordillo1'] = params.get('seccionbordillo1', 0.2) # ancho base bordillo
    seccionbordillo2 = params['seccionbordillo2'] = params.get('seccionbordillo2', 0.2) # ancho corona bordillo
    seccionbordillo3 = params['seccionbordillo3'] = params.get('seccionbordillo3', 0.3) # alto bordillo
    ecarpetaasf = params['ecarpetaasf'] = params.get('ecarpetaasf', 0.08) # espesor carperta asfaltica
    
    IM = params['IM'] = params.get('IM', 1.33)  # factor de amplificacion dinamica de carga
    n = params['n'] = params.get('n', 1) # Relacion modular
    b1 = params['b1'] = params.get('b1', 2.2) # Distancia para el calculo del factor de distribucion regla de la palanca
    b2 = params['b2'] = params.get('b2', 0.4) # Distancia para el calculo del factor de distribucion regla de la palanca
    rec = params['rec'] = params.get('rec', 0.1) # recubrimiento del acero
    rbarra = params['rbarra'] = params.get('rbarra', 8) # referencia de la barra para flexion
    rbarras = params['rbarras'] = params.get('rbarras', 3) # referencia de la barra para superficie
    rbarrae = params['rbarrae'] = params.get('rbarrae', 4) # referencia de la barra para estribos
    abarra = params['abarra'] = params.get('abarra', 0.000510) # area de la barra en metros para flexion
    abarras = params['abarras'] = params.get('abarras', 0.000071) # area de la barra en metros para superficie
    abarrae = params['abarrae'] = params.get('abarrae', 0.000129) # area de la barra en metros para estribos
    pb1 = params['pb1'] = params.get('pb1', 0.85) # valor para el calculo dela profundidad del bloque de compresiones
    duc = params['duc'] = params.get('duc', 0.003) # deformacion unitaria del concreto
    duas = params['duas'] = params.get('duas', 0.005) # deformacion unitaria del acero supuesta
    y3 = params['y3'] = params.get('y3', 0.75) # valor del concreto para el momento requerido por la combinacion de carga
    y1 = params['y1'] = params.get('y1', 1.6) # valor del concreto para el momento requerido por la combinacion de carga
    alpha = params['alpha'] = params.get('alpha', 90) # angulo para estribos verticales
   
    Ec = params['Ec'] = 4800 * (fc / 1000) ** 0.5 #Modulo de elasticidad del concreto en MPa
    hmin = params['hmin'] = 0.07*L # altura minima
    hseccion = params['hseccion'] = hviga + elosa  # altura de la seccion compuesta
    bf = params['bf'] = (svigas/2) + distvoladizo # ancho efectico de la aleta
    DClosa = params['DClosa'] = bf*elosa*pesoconcreto # carga de la losa
    DCviga = params['DCviga'] = baseviga*hviga*pesoconcreto # carga de la viga
    DCest = params['DCest'] = DClosa + DCviga # carga de la estructura
    DCbaranda = params['DCbaranda'] = (pesobaranda*nbarandas)/nb # carga de la baranda
    DCbordillo = params['DCbordillo'] = ((((seccionbordillo1+seccionbordillo2)/2)*seccionbordillo3)*pesoconcreto*nbordillo)/nb # carga del bordillo
    DCper = params['DCper'] = DClosa + DCviga + DCbordillo + DCbaranda # carga DC
    DW = params['DW'] = ecarpetaasf*bf*pesoasfalto # carga del asfalto   
    MDCest = params['MDCest'] = (DCest*(L**2))/8 # Momento maximo estructura
    MDW = params['MDW'] = (DW*(L**2))/8 # Momento maximo del asfalto
    MDCvol = params['MDCvol'] = ((DCbordillo + DCbaranda)*(L**2))/8 # Momento maximo del voladizo
    MDCper = params['MDCper'] = MDCest + MDCvol # Momento maximo carga DC
    VDCest = params['VDCest'] = (DCest*L)/2
    VDCvol = params['VDCvol'] = ((DCbordillo + DCbaranda)*L)/2
    VDCper = params['VDCper'] = VDCest + VDCvol

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
   
    de = params['de'] = distvoladizo - seccionbordillo1 # distancia entre eje de la viga exterior y la cara interna de la bordillo
    eg = params['eg'] = hseccion-(elosa/2)-(hviga/2) # distancia entre centroides de la viga y la losa
    kg = params['kg'] = n*(I + (A*(eg**2))) # Parametro para el calculo del factor de distribucion
    mg1i = params['mg1i'] = 0.06 + ((svigas/4.3)**0.4)*((svigas/L)**0.3)*((kg/(L*elosa**3))**0.1) # factor de distribucion para momento
    mg2i = params['mg2i'] = 0.075 + ((svigas/2.9)**0.6)*((svigas/L)**0.2)*((kg/(L*elosa**3))**0.1) # factor de distribucion para momento
    g1e = params['g1e'] =  (b1 + b2)/(2*svigas) # factor de distribucion sin mayorar por el factor de presencia multiple
    mg1e = params['mg1e'] = 1.2*g1e # factor de distribucion para momento
    mg2e = params['mg2e'] = (0.77 + (de/2.80))*mg2i # factor de distribucion para momento
    
    mg1ic = params['mg1ic'] = 0.36 + (svigas/7.6) # factor d7e distribucion para cortante
    mg2ic = params['mg2ic'] = 0.2 + (svigas/3.6)-((svigas/10)**2) # factor de distribucion para cortante
    mg1ec = params['mg1ec'] = 1.2*g1e # factor de distribucion para cortante
    mg2ec = params['mg2ec'] = (0.60 + (de/3))*mg2ic # factor de distribucion para cortante
    
    model = create_model(params)

    momentos_flectores_cargas_estructura(params, model)
    fuerzas_internas_cargas_permanentes(params, model)
    momentos_flectores_carga_viva_vehicular(params, model)
    combinaciones_carga(params, model)
    combinaciones_cargav(params, model)

    MLv = params['MLv'] = params['MLVmax'] # momento generado por el vehiculo
    MLc = params['MLc'] = params['MLCmax'] # momento generado por el carril
    
    MLLIM = params['MLLIM'] = (IM*MLv) + MLc # Momento maximo carga viva vehicular


    MLLIMp = params['MLLIMp'] = MLLIM*min(max(mg1i,mg2i),max(mg2e,mg1e)) # momento maximo debido a la carga viva con el factor de distribucion maximo hallado
    MUI = params['MUI'] = factormodcarga*((1.25*MDCper)+(1.5*MDW)+(1.75*MLLIMp)) # momento ultimo para resistencia I
    
    nbarra = params['nbarra'] = params.get('nbarra', 2)
    Sbarra = params['Sbarra'] = params.get('Sbarra', 0.25)

    As = params['As'] = nbarra*abarra
    rece = params['rece'] = rec + 0.03
    d = params['d'] = hseccion - rece  # altura efectiva
    p = params['p'] = As/(baseviga*d)
    a = params['a'] = (p*d*fy)/(0.85*fc) # posicion del eje neutro
    Mn = params['Mn'] = frf*As*fy*(d - (a/2))
    
    d2 = params['d2'] = d - Sbarra
    p2 = params['p2'] = As/(baseviga*d2)
    a2 = params['a2'] = (p2*d2*fy)/(0.85*fc) # posicion del eje neutro
    Mn2 = params['Mn2'] = frf*As*fy*(d2 - (a2/2))
    
    d3 = params['d3'] = d2 - Sbarra
    p3 = params['p3'] = As/(baseviga*d3)
    a3 = params['a3'] = (p3*d3*fy)/(0.85*fc) # posicion del eje neutro
    Mn3 = params['Mn3'] = frf*As*fy*(d3 - (a3/2))
    
    d4 = params['d4'] = d3 - Sbarra
    p4 = params['p4'] = As/(baseviga*d4)
    a4 = params['a4'] = (p4*d4*fy)/(0.85*fc) # posicion del eje neutro
    Mn4 = params['Mn4'] = frf*As*fy*(d4 - (a4/2))
    
    

    
    k = params['k'] = MUI/(bf*(d**2)) # parametro K para la cuantia
    m = params['m'] = fy/(0.85*fc) # parametro m para la cuantia
   
    # p = params['p'] = (1/m)*(1-(1-((2*m*k)/(frf*fy)))**0.5) # cuantia
   
    # As = params['As'] = p*d*bf # acero de refuerzo
    
    # nbarra = params['nbarra'] = np.ceil(As/abarra) # numero de barras a usar
    
    
    pc = params['pc'] = (As*fy)/(0.85*fc*bf*pb1) # profundidad del bloque de compresiones
    dua = params['dua'] = (d-pc)*(duc/pc) # deformacion unitaria del acero
    Ast = params['Ast'] = nbarra*abarra # area de acero total
    Askmin = params['Askmin'] = ((d*1000) - 760)   # area del acero de superficie minimo
    Askmax = params['Askmax'] = ((Ast*1000000)/4) # area del acero de superficie maximo
    Ssup = params['Ssup'] = d/6 # espaciamiento de barras superficiales
    dv1 = params['dv1'] = 0.9*d
    dv2 = params['dv2'] = 0.72*hseccion
    dv = params['dv'] = max(dv1,dv2)
    distc = params['distc'] = max(dv1,dv2)  + (aapoyo/2)

    VDCmax = params['VDCmax'] 
    VDW = params['VDW'] = params.get('VDW', 0)
    VLv = params['VLv'] = params['VLVmax']
    VLc = params['VLc'] = params['VLCmax']
    
    MUIdv = params['MUIdv'] = params.get('MUIdv', 625.66) # momento ultimo a cierta distancia del apoyo, debe ser calculado
    nbarradv = params['nbarradv'] = params.get('nbarradv', 4)

    VLLIM = params['VLLIM'] = (IM*VLv) + VLc # Cortante maximo carga viva vehicular
    

    VLLIMp = params['VLLIMp'] = VLLIM*min(max(mg1ic,mg2ic),max(mg2ec,mg1ec))  
    VUI = params['VUI'] = factormodcarga*((1.25*VDCmax)+(1.5*VDW)+(1.75*VLLIMp)) # cortante ultimo para resistencia I
    VN = params['VN'] = 0.25*fc*baseviga*dv
    vu = params['vu'] = VUI/(frf*baseviga*dv)
    duas = params['duas'] = ((MUIdv/dv)+VUI)/(Es*nbarradv*abarra)

    angulo1 = params['angulo1'] = 29 + (3500*duas)
    angulo2 = params['angulo2'] = 4.8/(1 + (750*duas))
    Vc = params['Vc'] = 0.083*angulo2*((fc/1000)**0.5)*baseviga*dv
    Vs = params['Vs'] = (VUI/frf) - (Vc*1000)

    pcv = params['pcv'] = (nbarradv*abarra*(fy/1000))/(0.85*(fc/1000)*bf) # profundidad del bloque de compresiones en la seccion critica por cortante
    dvc = params['dvc'] = d - (pcv/2)
    Av = params['Av'] = 2*abarrae
    S = params['S'] = (fy*dvc*Av*((np.cos(angulo1)/np.sin(angulo1))+(np.cos(alpha)/np.sin(alpha)))*np.sin(alpha))/Vs
    Avmin = params['Avmin'] = 0.083*baseviga*0.19*((fc/1000)**0.5)/(fy/1000)

    vuc = params['vuc'] = 0.125*fc
    Smax1 = params['Smax1'] = 0.8*dvc
    Smax2 = params['Smax2'] = 0.4*dvc
   




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
    # carpeta asfaltica
    # momentos_flectores_carpeta_asfaltica(superestructura, model)
    # momentos_flectores_bordillos_barandas(superestructura, model)
    # carga viva vehicular
    

    # combinaciones de carga

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

    return model

def plot(x, y, name, title, ylabel, no_ticks=2, invert_yaxis=False):
    # loadPattern = model.load_patterns['MDC']
    # mz = model.internal_forces[loadPattern][frame].mz

    fig, ax = plt.subplots()

    ax.plot(x, y, 'r')
    ax.set_title(title)  # 'Momento flector'
    ax.set_xlim(min(x), max(x))
    # ax.set_ylim(ymax=0)

    if invert_yaxis: ax.invert_yaxis()

    ax.set_xlabel('m')
    ax.set_ylabel(ylabel)  # 'kN m'

    ax.set_xticks(x[::no_ticks])
    ax.set_yticks(list(set(np.round_(y[::no_ticks], 3))))
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


    plot(x, mz, 'Mdnc', 'Momentos flectores', 'kN m')

    x = x[::2]
    mz = mz[::2]
    params['Mdnc'] = [[x, m] for x, m in zip(x, mz)]

def fuerzas_internas_cargas_permanentes(params, model):
    carga_permanente = params['DCper']  # superestructura['avaluoCarga']['cargaMuerta']['total']

    frame = model.frames[1]
    length = frame.get_length()
    loadPattern = model.add_load_pattern("cargaPermanente")
    loadPattern.add_distributed_load(frame.name, fy=-carga_permanente)

    model.solve_load_pattern(loadPattern.name)

    fy = frame.get_internal_forces(loadPattern.name, 40)['fy']
    mz = frame.get_internal_forces(loadPattern.name, 40)['mz'] # model.internal_forces[loadPattern.name][frame.name].mz
    x = np.linspace(0, length, len(mz))

    plot(x, fy, 'VDC', 'Fuerza cortante', 'kN', no_ticks=4)
    plot(x, mz, 'MDC', 'Momentos flectores', 'kN m', invert_yaxis=True, no_ticks=4)

    x = x[::4]
    fy = fy[::4]
    mz = mz[::4]

    params['VDC'] = [[x, v] for x, v in zip(x, fy)]
    params['MDC'] = [[x, m] for x, m in zip(x, mz)]

    params['VDCmax'] = max(fy)

    return params

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
    # viga
    frame = model.frames[1]
    length = frame.get_length()

    # camion
    x_ejes_camion = np.array([0] + camion_CC14['separacion_ejes'])
    for i in range(len(x_ejes_camion)):
        for j in range(i+1, len(x_ejes_camion)):
            x_ejes_camion[j] += x_ejes_camion[i]
    length_camion = x_ejes_camion[-1]
    peso_ejes_camion = camion_CC14['peso_ejes']
    
    # tandem
    x_ejes_tandem = np.array([0] + tandem_CC14['separacion_ejes'])
    for i in range(len(x_ejes_tandem)):
        for j in range(i+1, len(x_ejes_tandem)):
            x_ejes_tandem[j] += x_ejes_tandem[i]
    
    length_tandem = x_ejes_tandem[-1]
    peso_ejes_tandem = tandem_CC14['peso_ejes']

    # casos de carga
    casos_carga_camion_CC14 = []
    n = 401 # NUMERO PARADAS CAMION
    vehiculos = ['camion', 'tandem']
    length_vehiculos = [length_camion, length_tandem]
    x_ejes_vehiculos = [x_ejes_camion, x_ejes_tandem]
    peso_ejes_vehiculos = [peso_ejes_camion, peso_ejes_tandem]

    for vehiculo, length_vehiculo, x_ejes, peso_ejes in zip(vehiculos, length_vehiculos, x_ejes_vehiculos, peso_ejes_vehiculos):
        for i in range(n):
            x = (i / (n - 1)) * (length + length_vehiculo)

            loadPattern = model.add_load_pattern(f'{vehiculo}: {x:.3f} m')

            for j, xi in enumerate(x - x_ejes):
                if 0 < xi < length:
                    loadPattern.add_point_load_at_frame(frame.name, fy=(-peso_ejes[j], xi / length))

            casos_carga_camion_CC14.append(loadPattern.name)

            x = (n - 1 - i) / (n - 1) * (length + length_vehiculo) - length_vehiculo

            loadPattern = model.add_load_pattern(f'{vehiculo}: -{x:.3f} m')

            for j, xi in enumerate(x + x_ejes):
                if 0 < xi < length:
                    loadPattern.add_point_load_at_frame(frame.name, fy=(-peso_ejes[j], xi / length))

            casos_carga_camion_CC14.append(loadPattern.name)

    #  carril
    w = -camion_CC14['carga_carril']
    loadPattern = model.add_load_pattern('carril')
    loadPattern.add_distributed_load(frame.name, fy=w)

    model.solve()

    n = 401  # cantidad de valores por elemento por defecto
    cortantes = {'{:.3f}'.format(i / (n - 1) * length): [] for i in range(n)}
    momentos = {'{:.3f}'.format(i / (n - 1) * length): [] for i in range(n)}

    for load_pattern in casos_carga_camion_CC14:
        fy = frame.get_internal_forces(load_pattern, n-1)['fy']
        mz = frame.get_internal_forces(load_pattern, n-1)['mz']

        for i in range(n):
            cortantes['{:.3f}'.format(i / (n - 1) * length)].append(fy[i])
            momentos['{:.3f}'.format(i / (n - 1) * length)].append(mz[i])

    cortantes_maximos = []
    cortantes_minimos = []
    
    momentos_maximos = []
    for i, (v, m) in enumerate(zip(cortantes.values(), momentos.values())):
        cortantes_maximos.append([i / (n - 1) * length, max(v)])
        cortantes_minimos.append([i / (n - 1) * length, min(v)])
        momentos_maximos.append([i / (n - 1) * length, max(m)])

    loadPattern = model.load_patterns['carril']
    cortantes_carril = [[(i / (n - 1)) * length, v] for i, v in enumerate(frame.get_internal_forces(loadPattern.name, n-1)['fy'])]
    momentos_carril = [[(i / (n - 1)) * length, m] for i, m in enumerate(frame.get_internal_forces(loadPattern.name, n-1)['mz'])]

    cortantes_max_carga_vehicular = []
    cortantes_min_carga_vehicular = []
    momentos_carga_vehicular = []
    # factor_distribucion = params['factorDistribucionDiseno']
    for i, (v_min_camion, v_max_camion, v_carril, m_camion, m_carril) in enumerate(zip(cortantes_minimos, cortantes_maximos, cortantes_carril, momentos_maximos, momentos_carril)):
        x = m_camion[0]
        cortantes_max_carga_vehicular.append([x, 1.33 * v_max_camion[1] + v_carril[1]])
        cortantes_min_carga_vehicular.append([x, 1.33 * v_min_camion[1] + v_carril[1]])
        momentos_carga_vehicular.append([x, 1.33 * m_camion[1] + m_carril[1]])  # factor_distribucion * ()

    plot([x_m[0] for x_m in momentos_carga_vehicular], [x_m[1] for x_m in momentos_carga_vehicular], 'MLL', 'Momentos flectores', 'kN m', 40, True)

    plot([x_v[0] for x_v in cortantes_max_carga_vehicular], [x_v[1] for x_v in cortantes_max_carga_vehicular], 'VLL', 'Fuerza cortante', 'kN', 80)

    plot([x_v[0] for x_v in cortantes_min_carga_vehicular], [x_v[1] for x_v in cortantes_min_carga_vehicular], 'VLLmin', 'Fuerza cortante', 'kN', 80)
    
    params['MLV'] = momentos_maximos[::40]
    params['MLC'] = momentos_carril[::40]
    params['MLL'] = momentos_carga_vehicular[::40]

    params['VLV'] = cortantes_maximos[::40]
    params['VLC'] = cortantes_carril[::40]
    params['VLL_max'] = cortantes_max_carga_vehicular[::40]
    params['VLL_min'] = cortantes_min_carga_vehicular[::40]

    params['MLVmax'] = max([m[1] for m in momentos_maximos])
    params['MLCmax'] = max([m[1] for m in momentos_carril])
    params['MLLmax'] = max([m[1] for m in momentos_carga_vehicular])

    params['VLVmax'] = max([v[1] for v in cortantes_maximos])
    params['VLCmax'] = max([v[1] for v in cortantes_carril])
    params['VLLmax'] = max([v[1] for v in cortantes_max_carga_vehicular])

    return params
    

def combinaciones_carga(params, model):
    params['combinacionesCarga'] = {}

    length = params['L']
    n = 11

    m_carga_permanente = params['MDC']
    m_carga_vehicular = params['MLL']
    cargas = [m_carga_permanente, m_carga_vehicular]
    fd1 = params['mg1i']
    fd2 = params['mg2i']
    fd3 = params['mg1e']
    fd4 = params['mg2e']

    # estado límite de resistencia última
    mu = []
    for i, (carga_permanente, carga_vehicular) in enumerate(zip(*cargas)):
        mu.append([(i / (n - 1) * length), 1.25 * carga_permanente[1] + max(fd1,fd3) * 1.75 * carga_vehicular[1]])

    params['combinacionesCarga']['resistenciaUltima'] = mu

    # del mu

    # estado límite de resistencia IV
    # mu = []
    # for x, m in m_carga_permanente:
    #     mu.append([x, 1.5 * m])

    # superestructura['combinacionesCarga']['resistenciaIV'] = mu

    # estado límite servicio
    mu = []
    for i, (carga_permanente, carga_vehicular) in enumerate(zip(*cargas)):
        mu.append([(i / (n - 1) * length), 1 * carga_permanente[1]  + max(fd1, fd3) * 1 * carga_vehicular[1]])

    params['combinacionesCarga']['servicio'] = mu
      
    # print(mu)

def combinaciones_cargav(params, model):
    params['combinacionesCargav'] = {}

    length = params['L']
    n = 11

    m_carga_permanentev = params['VDC']
    m_carga_vehicularv = params['VLL_max']
    cargasv = [m_carga_permanentev, m_carga_vehicularv]
    fd1 = params['mg1i']
    fd3 = params['mg1e']

    #print(m_carga_permanentev)
    #print(m_carga_vehicularv)

    # estado límite de resistencia última
    vu = []
    for i, (carga_permanentev, carga_vehicularv) in enumerate(zip(*cargasv)):
        vu.append([(i / (n - 1) * length), 1.25 * carga_permanentev[1] + max(fd1, fd3) * 1.75 * carga_vehicularv[1]])

    params['combinacionesCargav']['resistenciaUltimav'] = vu

    # estado límite de resistencia IV
    # vu = []
    # for x, m in m_carga_permanentev:
    #    vu.append([x, 1.5 * m])

    # superestructura['combinacionesCarga']['resistenciaIV'] = vu

    # estado límite servicio
    vu = []
    for i, (carga_permanentev, carga_vehicularv) in enumerate(zip(*cargasv)):
        vu.append([(i / (n - 1) * length), 1 * carga_permanentev[1]  + max(fd1, fd3) * 1 * carga_vehicularv[1]])

    params['combinacionesCargav']['servicio'] = vu
      
    #print(mu)  

if __name__ == '__main__':
    superestructura = superestructura()
    # pp.pprint(superestructura)

    # doc template
    doc = DocxTemplate('template.docx')
    doc.render(superestructura)
    doc.save('output.docx')
    print('todo ok')
