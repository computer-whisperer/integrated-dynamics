from sympy.physics.mechanics import *
from sympy import symbols, Matrix, solve, lambdify
init_vprinting()

theta_x, theta_y, theta_z = dynamicsymbols("theta_x theta_y theta_z")
d_theta_x, d_theta_y, d_theta_z = dynamicsymbols("theta_x theta_y theta_z", 1)

omega_x, omega_y, omega_z = dynamicsymbols("omega_x omega_y omega_z")
d_omega_x, d_omega_y, d_omega_z = dynamicsymbols("omega_x omega_y omega_z", 1)

pos_x, pos_y, pos_z = dynamicsymbols("pos_x pos_y pos_z")
d_pos_x, d_pos_y, d_pos_z = dynamicsymbols("pos_x pos_y pos_z", 1)

vel_x, vel_y, vel_z = dynamicsymbols("vel_x vel_y vel_z")
d_vel_x, d_vel_y, d_vel_z = dynamicsymbols("vel_x vel_y vel_z", 1)

m, g, k, t = symbols("m g k t")
#m = 2
#g = 9.81
#k = 200
#t = symbols("t")

world_frame = ReferenceFrame("world_frame")
origin = Point("origin")
origin.set_vel(world_frame, 0)

rov_frame = world_frame.orientnew("rov_frame", "Body", (theta_x, theta_y, theta_z), "XYZ")
rov_frame.set_ang_vel(world_frame, world_frame.x*omega_x + world_frame.y*omega_y + world_frame.z*omega_z)

rov_pos = origin.locatenew("rov_pos", world_frame.x * pos_x + world_frame.y * pos_y + world_frame.z * pos_z)

link_pos = rov_pos.locatenew("link_pos", rov_frame.x * 2)
link_pos.set_vel(rov_frame, 0)


rov_inertia = inertia(rov_frame, m / 5**2, m / 5**2, m / 5**2)

rov_body = RigidBody("rov_particle", rov_pos, rov_frame, m, (rov_inertia, rov_pos))

kde = Matrix([
    d_theta_x - omega_x,
    d_theta_y - omega_y,
    d_theta_z - omega_z,
    d_pos_x - vel_x,
    d_pos_y - vel_y,
    d_pos_z - vel_z,
])


dq_dict = solve(kde, [d_theta_x, d_theta_y, d_theta_z, d_pos_x, d_pos_y, d_pos_z])

rov_pos.set_vel(world_frame, rov_pos.pos_from(origin).dt(world_frame).subs(dq_dict))
link_pos.v2pt_theory(rov_pos, world_frame, rov_frame)

f_c = Matrix([])
f_v = Matrix([])

forces = [(rov_pos, -g*m*world_frame.y), (link_pos, -pos_y*k*world_frame.y)]

KM = KanesMethod(world_frame,
                 q_ind=[pos_x, pos_y, pos_z, theta_x, theta_y, theta_z],
                 u_ind=[vel_x, vel_y, vel_z, omega_x, omega_y, omega_z],
                 kd_eqs=kde)
(fr, frstar) = KM.kanes_equations(forces, [rov_body])


MM = KM.mass_matrix_full
forcing = KM.forcing_full
rhs = MM.inv()*forcing

mm_inv = MM.inv()

vprint(MM)

vprint(forcing)

vprint(rhs)

ind_map = [pos_x, pos_y, pos_z, theta_x, theta_y, theta_z, vel_x, vel_y, vel_z, omega_x, omega_y, omega_z]

rhs_lamb = lambdify(ind_map, rhs.subs({g: 9.81, m: 10, k: 1000}))

print("\n\n\n ========== Simulation ============== \n")
states = {
    pos_x: 0,
    pos_y: 0,
    pos_z: 0,
    theta_x: 0,
    theta_y: 0,
    theta_z: 0,
    vel_x: 0,
    vel_y: 0,
    vel_z: 0,
    omega_x: 0,
    omega_y: 0,
    omega_z: 0
}
for _ in range(20):
    dx = rhs_lamb(*[states[symb] for symb in ind_map])*0.01
    for i in range(len(ind_map)):
        states[ind_map[i]] += dx[i, 0]
    mprint(states)


#MM = KM.mass_matrix
#forcing = KM.forcing
#rhs = MM.inv() * forcing
#kdd = KM.kindiffdict()
#rhs = rhs.subs(kdd)
#rhs.simplify()
#mprint(rhs)

#pos_operating_point = {
#    pos_x: 0,
#    pos_z: 0,
#    theta_x: 0,
#    theta_y: 0,
#    theta_z: 0
#}
#
#vel_operating_point = {
#    vel_x: 0,
#    vel_y: 0,
#    vel_z: 0,
#    omega_x: 0,
#    omega_y: 0,
#    omega_z: 0
#}
#
#acc_operating_point = {
#    d_vel_x: 0,
#    d_vel_y: 0,
#    d_vel_z: 0,
#    d_omega_x: 0,
#    d_omega_y: 0,
#    d_omega_z: 0
#}
#
#
#
#
#print("\n\n\n ========== Linearization ============ \n")
#A, B, r = KM.linearize(op_point=[pos_operating_point],
#                             A_and_B=True,
#                             new_method=True,
#                             simplify=True)
#
#vprint(A)
#vprint(B)
#vprint(r)

