from main import superestructura
from docxtpl import DocxTemplate
# from pyFEM import Structure

params = {
    'fc': 18.9e3,             # kPa
    # 'pesoconcreto': 24 kN
    'fy': 420000,
    'Es': 200000000,        # kPa
    'nb': 2,
    'nl': 1,
    'L' : 9.14,
    'aapoyo': 0.15,
    'svigas': 2.5,
    'distvoladizo': 1.1,
    'baseviga': 0.2,
    'hviga': 0.95,
    'elosa': 0.2,
    'nbarandas': 0,
    'nbordillo': 2,
    'seccionbordillo1': 0.25,
    'seccionbordillo2': 0.20,
    'seccionbordillo3': 0.23,
    'ecarpetaasf': 0,
    'b1': 2.75,
    'b2': 0.95,
    'rec': 0.05,
    'nbarra': 2,
    'rbarra': 8,
#     'rbarra':
}

params = superestructura(params)

doc = DocxTemplate('template.docx')
doc.render(params)
doc.save('puente_1.docx')

# def create_model(params={}):
#     material = {
#         'name': 'concreto',
#         'E': params['Ec']
#     }

#     section = {
#         'name': 'seccionCompuesta',
#         'Iz': params['Ic']
#     }

#     joint_a = {
#         'name': 'A',
#         'x': 0
#     }

#     joint_b = {
#         'name': 'B',
#         'x': params['L']
#     }

#     frame_1 = {
#         'name': 1,
#         'joint_j': 'A',
#         'joint_k': 'B',
#         'material': 'concreto',
#         'section': 'seccionCompuesta'
#     }

#     # print(material, section, joint_a, joint_b, frame_1)
    
#     # create model
#     model = Structure(uy=True, rz=True)

#     model.add_material(**material)
#     model.add_section(**section)
#     model.add_joint(**joint_a)
#     model.add_joint(**joint_b)
#     model.add_frame(**frame_1)

#     model.add_support('A',uy=True)
#     model.add_support('B',uy=True)

#     model.add_load_pattern('DC')
#     model.add_distributed_load('DC', 1, fy=-params['DCper'])

#     # for values in model.__dict__.values():
#     #     if isinstance(values, dict):
#     #         for value in values.values():
#     #             print(value)
    
#     model.solve()

    

# print(params)
# create_model(params)
