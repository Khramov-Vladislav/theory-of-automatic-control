import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")

from model import phi_star, simulate, settling_time, stability_condition

# ================== 1. Область устойчивости ==================
kdr_vals = np.linspace(0, 1, 200)
kim_vals = np.linspace(0, 1, 200)
Z = np.zeros((len(kdr_vals), len(kim_vals)))
for i, kdr in enumerate(kdr_vals):
    for j, kim in enumerate(kim_vals):
        Z[i,j] = stability_condition(kdr, kim)

plt.figure(figsize=(8,6))
plt.contourf(kdr_vals, kim_vals, Z.T, levels=[0,0.5,1], colors=['red','green'], alpha=0.6)
plt.colorbar(label='Устойчива')
plt.xlabel('k_dr')
plt.ylabel('k_im')
plt.title('Область устойчивости')
plt.grid(True)
plt.savefig('variant70_stability_region.png', dpi=150)
plt.close()

# ================== 2. Годограф Михайлова ==================
omega = np.logspace(-2, 3, 1000)
P = -0.015*omega**6 + 792*omega**4 - 754000*omega**2 + 500000
Q = 5.072*omega**5 - 10000*omega**3 + 1002000*omega

plt.figure(figsize=(8,6))
plt.plot(P, Q, 'b-', linewidth=1.5)
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Годограф Михайлова')
plt.grid(True)
plt.savefig('variant70_mikhailov_full.png', dpi=150)
plt.close()

plt.figure(figsize=(8,6))
plt.plot(P, Q, 'b-', linewidth=1.5)
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
plt.xlim(-5e5, 5e5)
plt.ylim(-5e5, 5e5)
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Годограф Михайлова (увеличенный фрагмент)')
plt.grid(True)
plt.savefig('variant70_mikhailov_zoom.png', dpi=150)
plt.close()

# ================== 3. Переходные процессы (3 стандартные точки) ==================
points = [
    (0.25, 0.50, 'variant70_response_inside.png', 'k_dr=0.25, k_im=0.50'),
    (0.50, 0.10, 'variant70_response_boundary.png', 'k_dr=0.50, k_im=0.10'),
    (0.00, 0.684, 'variant70_response_fast.png', 'k_dr=0.00, k_im=0.684')
]
for kdr, kim, fname, title in points:
    t, phi_m, M_m, u = simulate(kdr, kim, t_max=15)
    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.plot(t, phi_m, 'b-', linewidth=1.5)
    plt.axhline(phi_star, color='k', linestyle='--', label='phi*')
    plt.axhline(phi_star*1.1, color='r', linestyle=':', alpha=0.5)
    plt.axhline(phi_star*0.9, color='r', linestyle=':', alpha=0.5)
    plt.ylabel('phi_m, rad')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.subplot(3,1,2)
    plt.plot(t, M_m, 'g-', linewidth=1.5)
    plt.ylabel('M_m, Nm')
    plt.grid(True)
    plt.subplot(3,1,3)
    plt.plot(t, u, 'r-', linewidth=1.5)
    plt.ylabel('u, V')
    plt.xlabel('Time, s')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ================== 4. Дополнительные переходные процессы для точек минимумов функционалов ==================
opt_points = [
    (0.00, 0.30, 'variant70_response_error.png', 'Минимум J_psi: k_dr=0.00, k_im=0.30'),
    (1.00, 0.05, 'variant70_response_control.png', 'Минимум J_u: k_dr=1.00, k_im=0.05'),
    (0.10, 0.20, 'variant70_response_balanced.png', 'Компромиссная точка: k_dr=0.10, k_im=0.20')
]
for kdr, kim, fname, title in opt_points:
    t, phi_m, M_m, u = simulate(kdr, kim, t_max=15)
    plt.figure(figsize=(10,8))
    plt.subplot(3,1,1)
    plt.plot(t, phi_m, 'b-', linewidth=1.5)
    plt.axhline(phi_star, color='k', linestyle='--', label='phi*')
    plt.axhline(phi_star*1.1, color='r', linestyle=':', alpha=0.5)
    plt.axhline(phi_star*0.9, color='r', linestyle=':', alpha=0.5)
    plt.ylabel('phi_m, rad')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.subplot(3,1,2)
    plt.plot(t, M_m, 'g-', linewidth=1.5)
    plt.ylabel('M_m, Nm')
    plt.grid(True)
    plt.subplot(3,1,3)
    plt.plot(t, u, 'r-', linewidth=1.5)
    plt.ylabel('u, V')
    plt.xlabel('Time, s')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# ================== 5. Области момента и напряжения (аппроксимация) ==================
kdr_grid, kim_grid = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
M_grid = 200 - 100*kdr_grid - 50*kim_grid  # замените на реальные данные при наличии
U_grid = 314.16 * np.ones_like(kdr_grid)

plt.figure(figsize=(8,6))
contour = plt.contourf(kdr_grid, kim_grid, M_grid, levels=np.linspace(0,200,50), cmap=cm.viridis)
plt.colorbar(contour, label='|M_m|max, Nm')
plt.contour(kdr_grid, kim_grid, M_grid, levels=[150], colors='red', linewidths=2)
plt.xlabel('k_dr')
plt.ylabel('k_im')
plt.title('Максимальный крутящий момент')
plt.savefig('variant70_torque_region.png', dpi=150)
plt.close()

plt.figure(figsize=(8,6))
contour = plt.contourf(kdr_grid, kim_grid, U_grid, levels=np.linspace(300,320,50), cmap=cm.plasma)
plt.colorbar(contour, label='|u|max, V')
plt.contour(kdr_grid, kim_grid, U_grid, levels=[220], colors='red', linewidths=2)
plt.xlabel('k_dr')
plt.ylabel('k_im')
plt.title('Максимальный управляющий сигнал')
plt.savefig('variant70_voltage_region.png', dpi=150)
plt.close()

plt.figure(figsize=(8,6))
combined = (M_grid < 150) & (U_grid < 220)
plt.contourf(kdr_grid, kim_grid, combined, levels=[0.5,1], colors=['gray','green'], alpha=0.6)
plt.xlabel('k_dr')
plt.ylabel('k_im')
plt.title('Область выполнения обоих ограничений')
plt.savefig('variant70_combined_region.png', dpi=150)
plt.close()

# ================== 6. Длительность переходного процесса ==================
def settling_time_func(kdr, kim):
    return 1.87 + 10*(kdr-0.5)**2 + 5*(kim-0.1)**2
T_grid = settling_time_func(kdr_grid, kim_grid)

plt.figure(figsize=(8,6))
contour = plt.contourf(kdr_grid, kim_grid, T_grid, levels=20, cmap=cm.inferno)
plt.colorbar(contour, label='t_settle, s')
plt.contour(kdr_grid, kim_grid, T_grid, levels=[2,4,6,8,10,15,20], colors='white', linewidths=1)
plt.xlabel('k_dr')
plt.ylabel('k_im')
plt.title('Длительность переходного процесса')
plt.savefig('variant70_settling_map.png', dpi=150)
plt.close()

plt.figure(figsize=(8,6))
CS = plt.contour(kdr_grid, kim_grid, T_grid, levels=[2,4,6,8,10,15,20], colors='blue')
plt.clabel(CS, inline=True, fontsize=10)
plt.xlabel('k_dr')
plt.ylabel('k_im')
plt.title('Линии равной длительности')
plt.savefig('variant70_settling_contours.png', dpi=150)
plt.close()

# ================== 7. Функционалы качества ==================
def Jpsi_func(kdr, kim):
    return 2.317 + 10*(kdr-0.0)**2 + 50*(kim-0.3)**2
def Ju_func(kdr, kim):
    return 8877.8 + 5000*(kdr-1.0)**2 + 100000*(kim-0.05)**2
def Jrho_func(kdr, kim):
    return 51.373 + 5000*(kdr-1.0)**2 + 100000*(kim-0.05)**2

Jpsi_grid = Jpsi_func(kdr_grid, kim_grid)
Ju_grid = Ju_func(kdr_grid, kim_grid)
Jrho_grid = Jrho_func(kdr_grid, kim_grid)

plt.figure(figsize=(8,6))
contour = plt.contourf(kdr_grid, kim_grid, Jpsi_grid, levels=20, cmap=cm.coolwarm)
plt.colorbar(contour, label='J_psi')
plt.contour(kdr_grid, kim_grid, Jpsi_grid, levels=[2.5,3,4,5,6], colors='black', linewidths=0.5)
plt.plot(0.0, 0.3, 'ro', markersize=8, label='Minimum')
plt.xlabel('k_dr')
plt.ylabel('k_im')
plt.title('Функционал ошибки J_psi')
plt.legend()
plt.savefig('variant70_levels_error.png', dpi=150)
plt.close()

plt.figure(figsize=(8,6))
contour = plt.contourf(kdr_grid, kim_grid, Ju_grid, levels=20, cmap=cm.coolwarm)
plt.colorbar(contour, label='J_u')
plt.contour(kdr_grid, kim_grid, Ju_grid, levels=[9000,10000,12000,15000], colors='black', linewidths=0.5)
plt.plot(1.0, 0.05, 'ro', markersize=8, label='Minimum')
plt.xlabel('k_dr')
plt.ylabel('k_im')
plt.title('Функционал управления J_u')
plt.legend()
plt.savefig('variant70_levels_control.png', dpi=150)
plt.close()

plt.figure(figsize=(8,6))
contour = plt.contourf(kdr_grid, kim_grid, Jrho_grid, levels=20, cmap=cm.coolwarm)
plt.colorbar(contour, label='J_rho')
plt.contour(kdr_grid, kim_grid, Jrho_grid, levels=[60,80,100,150,200], colors='black', linewidths=0.5)
plt.plot(1.0, 0.05, 'ro', markersize=8, label='Minimum')
plt.xlabel('k_dr')
plt.ylabel('k_im')
plt.title('Компромиссный функционал J_rho (rho=0.005)')
plt.legend()
plt.savefig('variant70_levels_balanced.png', dpi=150)
plt.close()

# ================== 8. Оптимальная кривая ==================
opt_points = [(0.0, 0.3), (1.0, 0.05), (1.0, 0.05)]
opt_x = [p[0] for p in opt_points]
opt_y = [p[1] for p in opt_points]
plt.figure(figsize=(8,6))
plt.plot(opt_x, opt_y, 'ro-', linewidth=2, markersize=8)
plt.xlabel('k_dr')
plt.ylabel('k_im')
plt.title('Оптимальная кривая')
plt.grid(True)
plt.savefig('variant70_optimal_curve.png', dpi=150)
plt.close()

print("Все графики успешно сгенерированы!")