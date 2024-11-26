import torch


class DDPM():
    def __init__(self,
                 device,
                 n_steps: int = 1000,
                 min_beta: float = 0.00085,
                 max_beta: float = 0.012,
                 beta_schedule = 'scaled_linear',
                 ):
        if beta_schedule == 'linear':
            betas = torch.linspace(min_beta, max_beta, n_steps)
        elif beta_schedule == 'scaled_linear':
            betas = torch.linspace(min_beta**0.5, max_beta**0.5, n_steps, dtype=torch.float32) ** 2
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product

        self.n_steps = n_steps
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bars = alpha_bars.to(device)

    def sample_forward(self, x, t, eps=None):
        self.alpha_bars = self.alpha_bars.to(x.device)
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1).to(x.device)
        if eps is None:
            eps = torch.randn_like(x).to(x.device)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res, eps

    def sample_backward_step(self, x_t, t, eps, simple_var=True):
        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)
        mean = (x_t -
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise
        return x_t