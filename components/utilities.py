import math


def rot_pos_vector(vector, theta):
    sin = math.sin(theta)
    cos = math.cos(theta)
    return [
        cos*vector[0] + sin*vector[1],
        cos*vector[1] + sin*vector[0],
        vector[2]+theta
    ]


def rot_vel_vector(vector, theta):
    sin = math.sin(theta)
    cos = math.cos(theta)
    return [
        cos*vector[0] + sin*vector[1],
        cos*vector[1] + sin*vector[0],
        vector[2]
    ]


def sub_vector(vector1, vector2):
    return [comp1 - comp2 for comp1, comp2 in zip(vector1, vector2)]


def add_vector(vector1, vector2):
    return [comp1 + comp2 for comp1, comp2 in zip(vector1, vector2)]