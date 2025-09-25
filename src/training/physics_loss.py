import torch

def navier_cauchy_residual(displacement, coords, youngs_modulus, poissons_ratio):
    """
    Calculates the residual of the Navier-Cauchy equation for static equilibrium.
    This function forms the core of the physics-informed loss. It uses automatic
    differentiation to compute the derivatives of the neural network's output
    with respect to the input coordinates.

    The equation is: μ∇²u + (λ+μ)∇(∇·u) = 0 (assuming no body forces)
    
    Args:
        displacement (torch.Tensor): The network's output displacement field [N, 3].
        coords (torch.Tensor): The input coordinates [N, 3], requires_grad=True.
        youngs_modulus (float or torch.Tensor): E [1] or [B]
        poissons_ratio (float or torch.Tensor): ν [1] or [B]
    
    Returns:
        torch.Tensor: The PDE residual at each coordinate [N, 3].
    """
    # ### --- CHANGE --- ###
    # Lamé parameters are now computed from the input arguments, not hardcoded.
    lmbda = (youngs_modulus * poissons_ratio) / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    mu = youngs_modulus / (2 * (1 + poissons_ratio))
    
    # Reshape for broadcasting to each node
    lmbda = lmbda.view(-1, 1)
    mu = mu.view(-1, 1)

    u, v, w = displacement[:, 0], displacement[:, 1], displacement[:, 2]

    # First derivatives
    # torch.autograd.grad computes gradients of a scalar sum w.r.t. inputs.
    # This is how we get du/dx, du/dy, etc.
    grad_outputs = torch.ones_like(u) # Create a tensor of ones for scalar projection
    grad_u = torch.autograd.grad(u, coords, grad_outputs=grad_outputs, create_graph=True)[0]
    grad_v = torch.autograd.grad(v, coords, grad_outputs=grad_outputs, create_graph=True)[0]
    grad_w = torch.autograd.grad(w, coords, grad_outputs=grad_outputs, create_graph=True)[0]
    
    du_dx, du_dy, du_dz = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
    dv_dx, dv_dy, dv_dz = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2]
    dw_dx, dw_dy, dw_dz = grad_w[:, 0], grad_w[:, 1], grad_w[:, 2]

    # Divergence of displacement field (a scalar value per node)
    div_u = du_dx + dv_dy + dw_dz

    # Gradient of the divergence (a vector per node)
    grad_div_u = torch.autograd.grad(div_u, coords, grad_outputs=grad_outputs, create_graph=True)[0]

    # Second derivatives (for Laplacian)
    # The laplacian is the sum of second partial derivatives: ∇²u = d²u/dx² + d²u/dy² + d²u/dz²
    du_dxx = torch.autograd.grad(du_dx, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 0]
    du_dyy = torch.autograd.grad(du_dy, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 1]
    du_dzz = torch.autograd.grad(du_dz, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 2]
    laplacian_u = du_dxx + du_dyy + du_dzz

    dv_dxx = torch.autograd.grad(dv_dx, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 0]
    dv_dyy = torch.autograd.grad(dv_dy, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 1]
    dv_dzz = torch.autograd.grad(dv_dz, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 2]
    laplacian_v = dv_dxx + dv_dyy + dv_dzz

    dw_dxx = torch.autograd.grad(dw_dx, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 0]
    dw_dyy = torch.autograd.grad(dw_dy, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 1]
    dw_dzz = torch.autograd.grad(dw_dz, coords, grad_outputs=grad_outputs, create_graph=True)[0][:, 2]
    laplacian_w = dw_dxx + dw_dyy + dw_dzz

    # Assemble the Navier-Cauchy equation components
    # Residual = Forcing Term. Since we have no body forces, the residual should be zero.
    residual_x = mu * laplacian_u.unsqueeze(1) + (lmbda + mu) * grad_div_u[:, 0].unsqueeze(1)
    residual_y = mu * laplacian_v.unsqueeze(1) + (lmbda + mu) * grad_div_u[:, 1].unsqueeze(1)
    residual_z = mu * laplacian_w.unsqueeze(1) + (lmbda + mu) * grad_div_u[:, 2].unsqueeze(1)
    
    # Using .squeeze() at the end to ensure correct shape
    return torch.cat([residual_x, residual_y, residual_z], dim=1)

def pde_loss(displacement, coords, youngs_modulus, poissons_ratio, batch):
    """ 
    Computes the mean squared error of the PDE residual. 
    This is the value that gets added to the main loss function.
    """
    # Expand material properties to match the number of nodes in each graph of the batch
    E_per_node = youngs_modulus[batch]
    nu_per_node = poissons_ratio[batch]
    
    residual = navier_cauchy_residual(displacement, coords, E_per_node, nu_per_node)
    # The loss is the mean of the squared residual vectors.
    return residual.pow(2).mean().to(torch.float32)