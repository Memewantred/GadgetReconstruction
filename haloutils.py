import cython
import numpy as np
import astropy
import astropy.units as u

# 参考：https://stackoverflow.com/questions/24733185/volume-of-convex-hull-with-qhull-from-scipy 关于使用scipy.spatial.ConvexHull计算体积的讨论
# 其四面体体积的计算方法来源于三个不共面向量的混合积等于平行六面体体积的公式
# 参考：https://max.book118.com/html/2020/0404/6152212035002153.shtm
def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6



# DECREPATED
def _a_dot(a, h0, om_m, om_l):
    om_k = 1.0 - om_m - om_l
    return h0 * a * np.sqrt(om_m * (a ** -3) + om_k * (a ** -2) + om_l)

def rho_crit(z, OmegaM0, OmegaL0, h):
    a = 1.0 / (1.0 + z)
    H_z = _a_dot(a, h, OmegaM0, OmegaL0) / a
    H_z = H_z * u.Unit('100 km/s/Mpc')
    rho_crit = 3 * H_z**2 / (8 * np.pi * astropy.constants.G)
    return rho_crit.to(u.kg / u.Mpc**3)

def virial_radius(snapshot, halo, overden=200, r_max=None, rho_def='particle', OmegaM0=None, OmegaL0=None, h=None, N_max=1e5):
    """
    
    """
    OmegaM0 = snapshot["header"]["Omega0"] if OmegaM0 is None else OmegaM0
    OmegaL0 = snapshot["header"]["OmegaLambda"] if OmegaL0 is None else OmegaL0
    h = snapshot["header"]["HubbleParam"] if h is None else h
    
    try:
        # Use Halolen to define radius
        halo_len = halo["GroupLen"]
    except KeyError:
        if rho_def == 'particle':
            rho_c = len(snapshot["pos"]) / snapshot["header"]["BoxSize"]**3
        elif rho_def == 'matter':
            z = 1. / snapshot["header"]["Time"] - 1
            rho_c = OmegaM0 * rho_crit(z, OmegaM0, OmegaL0, h) * (1 + z)**3
        threshold = overden * rho_c
    