import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



def f(x, y):
    return (x - 3) ** 2 + (y + 2) ** 2


def grad_f(x, y):
    return np.array([2 * (x - 3), 2 * (y + 2)], dtype=float)


def gradient_descent(x0, y0, alpha, max_iters=200, tol=1e-8):
    """
    Returns:
        path: (n,2) array of points (x,y)
        values: (n,) array of f(x,y)
    """
    x, y = float(x0), float(y0)
    path = [(x, y)]
    values = [f(x, y)]

    for _ in range(max_iters):
        g = grad_f(x, y)

        # update
        x_new = x - alpha * g[0]
        y_new = y - alpha * g[1]

        path.append((x_new, y_new))
        values.append(f(x_new, y_new))

        # stopping criteria
        if np.linalg.norm(g) < tol or np.hypot(x_new - x, y_new - y) < tol:
            break

        x, y = x_new, y_new

    return np.array(path), np.array(values)


def make_contour_plot(path, title, save_path, xlim=(-2, 8), ylim=(-8, 4), levels=30):
    xs = np.linspace(xlim[0], xlim[1], 400)
    ys = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    plt.figure(figsize=(7, 6))
    plt.contour(X, Y, Z, levels=levels)

    
    plt.plot(path[:, 0], path[:, 1], marker='o', linewidth=1.5, markersize=3)

    
    plt.scatter([3], [-2], marker='x', s=120)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def make_surface_plot(path, title, save_path, xlim=(-2, 8), ylim=(-8, 4)):
    from mpl_toolkits.mplot3d import Axes3D  

    xs = np.linspace(xlim[0], xlim[1], 200)
    ys = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0, alpha=0.7)

    
    ax.plot(path[:, 0], path[:, 1], f(path[:, 0], path[:, 1]),
            marker='o', linewidth=2, markersize=3)

    
    ax.scatter([3], [-2], [0], marker='x', s=120)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()



def main():
    
    base_dir = Path(__file__).parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    
    initial_points = [(0, 0), (5, -5)]
    learning_rates = [0.05, 0.1, 0.5]

    print("=== Gradient Descent HW4 ===")
    print("Function minimum at (3, -2)\n")

    
    for (x0, y0) in initial_points:
        for alpha in learning_rates:
            path, vals = gradient_descent(x0, y0, alpha, max_iters=200)

            xf, yf = path[-1]
            iters = len(path) - 1

            print(f"start=({x0},{y0}) | alpha={alpha} | iters={iters} "
                  f"| final=({xf:.6f},{yf:.6f}) | f={vals[-1]:.3e}")

             
            alpha_name = str(alpha).replace(".", "p")
            save_path = plots_dir / f"contour_start_{x0}_{y0}_alpha_{alpha_name}.png"

            title = f"Contour: start=({x0},{y0}), alpha={alpha}, iters={iters}"
            make_contour_plot(path, title, save_path)

    
    for (x0, y0) in initial_points:
        alpha = 0.1
        path, _ = gradient_descent(x0, y0, alpha, max_iters=200)
        save_path = plots_dir / f"surface_start_{x0}_{y0}_alpha_0p1.png"
        title = f"3D Surface: start=({x0},{y0}), alpha={alpha}"
        make_surface_plot(path, title, save_path)

    print("\nDone")
    print(f"Plots saved in: {plots_dir}")


if __name__ == "__main__":
    main()
