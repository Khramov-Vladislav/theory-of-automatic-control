import numpy as np
from scipy.integrate import solve_ivp

i = 50
Jm = 1.0
c = 1000.0
b = 4.0
Jd = 2e-3
s = 0.2
tau = 0.003
r = 0.2
k = 100.0
kpr = 1.0
phi_star = np.pi

Jd_pr = Jd * i**2
s_pr = s * i**2
r_pr = r * i

def system(t, x, kdr, kim):
    phi_rpr, dphi_rpr, phi_m, dphi_m, M_dpr, int_err = x
    u = -k * (kpr * (phi_rpr - phi_star) + kdr * dphi_rpr + kim * int_err)
    dx1 = dphi_rpr
    dx2 = (M_dpr - b*(dphi_rpr - dphi_m) - c*(phi_rpr - phi_m)) / Jd_pr
    dx3 = dphi_m
    dx4 = b*(dphi_rpr - dphi_m) + c*(phi_rpr - phi_m)
    dx5 = (-M_dpr - s_pr * dphi_rpr + r_pr * u) / tau
    dx6 = phi_m - phi_star
    return [dx1, dx2, dx3, dx4, dx5, dx6]

def simulate(kdr, kim, t_max=20):
    sol = solve_ivp(system, (0, t_max), [0,0,0,0,0,0],
                    args=(kdr, kim), method='RK45', rtol=1e-6, atol=1e-6)
    t = sol.t
    phi_m = sol.y[2]
    dphi_rpr = sol.y[1]
    phi_rpr = sol.y[0]
    dphi_m = sol.y[3]
    int_err = sol.y[5]
    u = -k * (kpr*(phi_rpr - phi_star) + kdr*dphi_rpr + kim*int_err)
    M_m = c*(phi_rpr - phi_m) + b*(dphi_rpr - dphi_m)
    return t, phi_m, M_m, u

def settling_time(t, phi_m, tol=0.1):
    threshold = tol * phi_star
    error = np.abs(phi_m - phi_star)
    for i in range(len(t)-1, -1, -1):
        if error[i] > threshold:
            return t[i+1] if i+1 < len(t) else t[-1]
    return t[0]

def stability_condition(kdr, kim):
    a6 = 3/200
    a5 = 634/125
    a4 = 1000*kdr + 542
    a3 = 4000*kdr + 9000
    a2 = 10**6*kdr + 504000
    a1 = 4000*kim + 10**6
    a0 = 10**6*kim
    if any(x <= 0 for x in [a0,a1,a2,a3,a4,a5,a6]):
        return False
    D1 = a5
    D2 = a5*a4 - a6*a3
    D3 = a4*(a3*a2 - a1*a4) - a2*(a5*a2) + a0*a4*a5
    D4 = a3*D3 - a1*(a4*(a3*a2 - a1*a4) - a2*(a5*a2)) + a0*a3*(a5*a4 - a6*a3)
    return (D1>0) and (D2>0) and (D3>0) and (D4>0) and (a0>0)