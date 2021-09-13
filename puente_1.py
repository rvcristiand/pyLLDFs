from main import superestructura
from pyFEM import Structure

params = {
    'nb': 2
}

params = superestructura(params)


def create_model(params={}):
    material = {
        'name': 'concreto',
        'E': params['Ec']
    }

    section = {
        'name': 'seccionCompuesta',
        'Iz': params['Ic']
    }

    joint_a = {
        'name': 'A',
        'x': 0
    }

    joint_b = {
        'name': 'B',
        'x': params['L']
    }

    frame_1 = {
        'name': 1,
        'joint_j': 'A',
        'joint_k': 'B',
        'material': 'concreto',
        'section': 'seccionCompuesta'
    }

    # print(material, section, joint_a, joint_b, frame_1)
    
    # create model
    model = Structure(uy=True, rz=True)

    model.add_material(**material)
    model.add_section(**section)
    model.add_joint(**joint_a)
    model.add_joint(**joint_b)
    model.add_frame(**frame_1)

    model.add_support('A',uy=True)
    model.add_support('B',uy=True)

    model.add_load_pattern('DC')
    model.add_distributed_load('DC', 1, fy=-params['DCper'])

    # for values in model.__dict__.values():
    #     if isinstance(values, dict):
    #         for value in values.values():
    #             print(value)
    
    model.solve()

    

# print(params)
create_model(params)
