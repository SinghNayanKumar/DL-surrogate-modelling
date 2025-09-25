import torch

def navier_cauchy_residual(displacement, coords, youngs_modulus, poissons_ratio):
    """
    Calculates the residual of the Navier-Cauchy equation for static equilibrium.
    The equation is: μ∇²u + (λ+μ)∇(∇·u) = 0 (assuming no body forces)
    
    Args:
        displacement (torch.Tensor): The network's output displacement field [N, 3].
        coords (torch.Tensor): The input coordinates [N, 3], requires_grad=True.
        youngs_modulus (float): E
        poissons_ratio (float): ν
    
    Returns:
        torch.Tensor: The PDE residual at each coordinate [N, 3].
    """
    # Lamé parameters
    lmbda = (youngs_modulus * poissons_ratio) / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    mu = youngs_modulus / (2 * (1 + poissons_ratio))

    u, v, w = displacement[:, 0], displacement[:, 1], displacement[:, 2]

    # First derivatives
    grad_u = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
    grad_v = torch.autograd.grad(v.sum(), coords, create_graph=True)[0]
    grad_w = torch.autograd.grad(w.sum(), coords, create_graph=True)[0]
    
    du_dx, du_dy, du_dz = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
    dv_dx, dv_dy, dv_dz = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2]
    dw_dx, dw_dy, dw_dz = grad_w[:, 0], grad_w[:, 1], grad_w[:, 2]

    # Divergence of displacement
    div_u = du_dx + dv_dy + dw_dz

    # Gradient of the divergence
    grad_div_u = torch.autograd.grad(div_u.sum(), coords, create_graph=True)[0]

    # Second derivatives (for Laplacian)
    du_dxx = torch.autograd.grad(du_dx.sum(), coords, create_graph=True)[0][:, 0]
    du_dyy = torch.autograd.grad(du_dy.sum(), coords, create_graph=True)[0][:, 1]
    du_dzz = torch.autograd.grad(du_dz.sum(), coords, create_graph=True)[0][:, 2]
    laplacian_u = du_dxx + du_dyy + du_dzz

    dv_dxx = torch.autograd.grad(dv_dx.sum(), coords, create_graph=True)[0][:, 0]
    dv_dyy = torch.autograd.grad(dv_dy.sum(), coords, create_graph=True)[0][:, 1]
    dv_dzz = torch.autograd.grad(dv_dz.sum(), coords, create_graph=True)[0][:, 2]
    laplacian_v = dv_dxx + dv_dyy + dv_dzz

    dw_dxx = torch.autograd.grad(dw_dx.sum(), coords, create_graph=True)[0][:, 0]
    dw_dyy = torch.autograd.grad(dw_dy.sum(), coords, create_graph=True)[0][:, 1]
    dw_dzz = torch.autograd.grad(dw_dz.sum(), coords, create_graph=True)[0][:, 2]
    laplacian_w = dw_dxx + dw_dyy + dw_dzz

    # Navier-Cauchy equation components
    residual_x = mu * laplacian_u + (lmbda + mu) * grad_div_u[:, 0]
    residual_y = mu * laplacian_v + (lmbda + mu) * grad_div_u[:, 1]
    residual_z = mu * laplacian_w + (lmbda + mu) * grad_div_u[:, 2]

    return torch.stack([residual_x, residual_y, residual_z], dim=1)

def pde_loss(displacement, coords, youngs_modulus=2.1e11, poissons_ratio=0.3):
    """ Computes the mean squared error of the PDE residual. """
    residual = navier_cauchy_residual(displacement, coords, youngs_modulus, poissons_ratio)
    return residual.pow(2).mean()