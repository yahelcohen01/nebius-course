import torch
import numpy as np
import matplotlib.pyplot as plt


def bowl(theta):
    x, y = theta[..., 0], theta[..., 1]
    return x**2 + 2*y**2


def camel(theta):
    x, y = theta[..., 0], theta[..., 1]
    return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2


def gradient_descent(f, theta0, lr=0.001, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()
        with torch.no_grad():
            theta -= lr * theta.grad
        theta.grad.zero_()
        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values


def momentum(f, theta0, lr=0.001, beta=0.9, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    v = torch.zeros_like(theta)
    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()
        with torch.no_grad():
            v = beta * v + theta.grad
            theta -= lr * v
        theta.grad.zero_()
        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values


def adagrad(f, theta0, lr=0.1, eps=1e-8, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    G = torch.zeros_like(theta)
    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()
        with torch.no_grad():
            G = G + theta.grad * theta.grad
            theta -= lr * theta.grad / (torch.sqrt(G) + eps)
        theta.grad.zero_()
        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values


def adam(f, theta0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=2000):
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(1, n_steps + 1):
        loss = f(theta)
        loss.backward()
        with torch.no_grad():
            m = beta1 * m + (1 - beta1) * theta.grad
            v = beta2 * v + (1 - beta2) * theta.grad * theta.grad
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            theta -= lr * m_hat / (torch.sqrt(v_hat) + eps)
        theta.grad.zero_()
        trajectory.append(theta.detach().clone())
        values.append(loss.item())

    return torch.stack(trajectory), values


def plot_trajectories(f, results, xlim=(-3, 3), ylim=(-2, 2), title="Optimization Trajectories", use_log=False):
    x_values = np.linspace(xlim[0], xlim[1], 400)
    y_values = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x_values, y_values)
    grid = np.stack((X, Y), axis=-1)
    Z = f(torch.tensor(grid, dtype=torch.float32)).detach().numpy()

    plt.figure(figsize=(10, 7))
    if use_log:
        plt.contour(X, Y, np.log1p(Z - Z.min()), levels=40, cmap='viridis', alpha=0.6)
    else:
        plt.contour(X, Y, Z, levels=40, cmap='viridis', alpha=0.6)

    colors = {'GD': '#B83A1F', 'Momentum': '#4F6B57', 'AdaGrad': '#C68A1F', 'Adam': '#2E3A5C'}
    for name, (trajectory, _) in results.items():
        traj = trajectory.numpy()
        plt.plot(traj[:, 0], traj[:, 1], '-', linewidth=1.5, label=name, color=colors.get(name, 'black'), alpha=0.85)
        plt.plot(traj[0, 0], traj[0, 1], 'o', color=colors.get(name, 'black'), markersize=8)
        plt.plot(traj[-1, 0], traj[-1, 1], '*', color=colors.get(name, 'black'), markersize=14)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_values(results, title="Function Value vs Iteration"):
    plt.figure(figsize=(10, 6))
    colors = {'GD': '#B83A1F', 'Momentum': '#4F6B57', 'AdaGrad': '#C68A1F', 'Adam': '#2E3A5C'}
    for name, (_, values) in results.items():
        plt.plot(values, linewidth=1.8, label=name, color=colors.get(name, 'black'), alpha=0.85)
    plt.xlabel('Iteration')
    plt.ylabel('f(x, y)')
    plt.title(title)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


def print_summary(results, f_name):
    print(f"\n{'='*60}")
    print(f"  {f_name} — Final Results")
    print(f"{'='*60}")
    print(f"{'Optimizer':<12} {'Final f(x,y)':>14} {'Final (x, y)'}")
    print(f"{'-'*60}")
    for name, (trajectory, values) in results.items():
        final = trajectory[-1].numpy()
        print(f"{name:<12} {values[-1]:>14.6f}   ({final[0]:.4f}, {final[1]:.4f})")


theta0 = [-1.5, 1.5]

results_bowl = {
    "GD":       gradient_descent(bowl, theta0, lr=0.05),
    "Momentum": momentum(bowl, theta0, lr=0.05, beta=0.9),
    "AdaGrad":  adagrad(bowl, theta0, lr=0.5),
    "Adam":     adam(bowl, theta0, lr=0.05),
}

print_summary(results_bowl, "Convex Bowl: f(x,y) = x² + 2y²")
plot_values(results_bowl, title="Convex Bowl — Function Value vs Iteration")
plot_trajectories(bowl, results_bowl, xlim=(-3, 3), ylim=(-2, 2), title="Convex Bowl — Optimization Trajectories")

theta0_camel = [-2.0, -1.5]

results_camel = {
    "GD":       gradient_descent(camel, theta0_camel, lr=0.01),
    "Momentum": momentum(camel, theta0_camel, lr=0.01, beta=0.9),
    "AdaGrad":  adagrad(camel, theta0_camel, lr=0.1),
    "Adam":     adam(camel, theta0_camel, lr=0.01),
}

print_summary(results_camel, "Six-Hump Camel")
plot_values(results_camel, title="Six-Hump Camel — Function Value vs Iteration")
plot_trajectories(camel, results_camel, xlim=(-3, 3), ylim=(-2, 2), title="Six-Hump Camel — Optimization Trajectories", use_log=True)
