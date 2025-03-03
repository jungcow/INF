import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_and_translation_to_matrix(euler_angles, translation):
    rotation_matrix = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    
    transformation_matrix = np.eye(4, dtype=np.float32)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation
    
    return transformation_matrix

if __name__ == '__main__':
    euler_rotation = [-95.1130765295202, 0.4133867468755132, -87.6530074235948] # e.g.: [roll, pitch, yaw] (unit: degree)
    translation = [0.8059487342834473, 0.266098290681839, -0.2622472643852234]          # x, y, z translation
    
    trans_mat = euler_and_translation_to_matrix(euler_rotation, translation)
    c2l = np.linalg.inv(trans_mat)
    print("Transformation Matrix:")
    print(c2l)
