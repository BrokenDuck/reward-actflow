from typing import Any, Optional, List
import numpy as np
import torch
import gpytorch


# ---------------------------------------------------------------------------
# Kernel implementations
# ---------------------------------------------------------------------------

KERNEL_CHOICES = ('cosine', 'rbf', 'linear')


class CosineKernel(gpytorch.kernels.Kernel):
    """Cosine similarity kernel for high-dimensional embeddings.

    Computes  k(x_i, x_j) = (x_i · x_j) / (||x_i|| ||x_j||)

    This is much better behaved than an RBF kernel in high-D because
    it only depends on the *direction* of the embedding vectors, not
    their magnitude.  On L2-normalised inputs it reduces to a simple
    dot-product kernel.

    The kernel values lie in [-1, 1].  To guarantee a positive-
    semi-definite kernel matrix (required by the GP) we shift to
    [0, 1] via  k' = (k + 1) / 2.
    """

    has_lengthscale = False  # no lengthscale parameter

    def forward(self, x1, x2, diag=False, **params):
        # L2 normalise (safe even if inputs are already normalised)
        x1_norm = x1 / x1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        x2_norm = x2 / x2.norm(dim=-1, keepdim=True).clamp(min=1e-9)

        if diag:
            # Diagonal only: row-wise dot products
            cos_sim = (x1_norm * x2_norm).sum(dim=-1)
        else:
            cos_sim = x1_norm @ x2_norm.transpose(-2, -1)

        # Shift from [-1, 1] to [0, 1] for PSD guarantee
        return (cos_sim + 1.0) / 2.0


class LinearKernel(gpytorch.kernels.Kernel):
    """Linear (dot-product) kernel:  k(x, x') = x · x' + c

    On L2-normalised inputs this is equivalent to the unshifted cosine
    kernel (values in [-1, 1]) plus a constant *c*.  The constant
    ensures PSD-ness when *c* >= 0.  We fix c = 1 so that k >= 0.

    The linear kernel measures alignment in the raw embedding space and
    scales with input magnitude (unlike cosine).  It is the simplest
    kernel — equivalent to Bayesian linear regression — and works best
    when the reward surface is approximately linear in embedding space.
    """

    has_lengthscale = False

    def __init__(self, variance: float = 1.0, offset: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('offset', torch.tensor(offset))
        # output variance is handled by the wrapping ScaleKernel, but we
        # keep a fixed positive offset for PSD guarantee.

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return (x1 * x2).sum(dim=-1) + self.offset
        return x1 @ x2.transpose(-2, -1) + self.offset


def build_kernel(name: str, lengthscale: float = 0.1, feat_dim: int = 768):
    """Construct a gpytorch kernel by name.

    Parameters
    ----------
    name : str
        One of ``'cosine'``, ``'rbf'``, or ``'linear'``.
    lengthscale : float
        Initial lengthscale for the RBF kernel (ignored by others).
    feat_dim : int
        Embedding dimensionality.  Used to set a sensible default
        lengthscale for RBF in high-D (``sqrt(D)`` heuristic).

    Returns
    -------
    gpytorch.kernels.Kernel
        Wrapped in a ``ScaleKernel`` so the GP can learn output variance.
    """
    if name == 'cosine':
        base = CosineKernel()
    elif name == 'rbf':
        base = gpytorch.kernels.RBFKernel()
        # In high-D, a reasonable default lengthscale is ~ sqrt(D) so
        # that exp(-||x-x'||^2 / 2l^2) doesn't collapse to 0 for every
        # pair.  Allow the user to override via the lengthscale arg.
        if lengthscale > 0:
            base.lengthscale = lengthscale
        else:
            base.lengthscale = float(feat_dim) ** 0.5
    elif name == 'linear':
        base = LinearKernel()
    else:
        raise ValueError(
            f"Unknown GP kernel '{name}'. Choose from {KERNEL_CHOICES}."
        )
    return gpytorch.kernels.ScaleKernel(base)


class GPModel(gpytorch.models.ExactGP):
    """Gaussian Process model for uncertainty estimation.

    Supports multiple kernel functions via the *covar_module* argument.
    """

    def __init__(self, train_x, train_y, likelihood,
                 covar_module: Optional[gpytorch.kernels.Kernel] = None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        # Default to cosine if no kernel supplied
        self.covar_module = covar_module or gpytorch.kernels.ScaleKernel(CosineKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPUncertaintyReward:
    """
    GP-based uncertainty reward for peptide generation.

    Uses embeddings from the diffusion model's backbone to estimate
    uncertainty over the sequence space for active learning / exploration.

    The GP uses a **cosine similarity kernel** which is well-behaved for
    the high-dimensional (768-D) backbone embeddings.  Embeddings are
    L2-normalised before being stored so the kernel reduces to a simple
    dot-product on the unit hypersphere.
    
    Parameters
    ----------
    diffusion_model : Diffusion
        The diffusion model to extract embeddings from
    device : torch.device
        Device to run computations on
    normalize_uncertainty : bool
        Whether to normalize uncertainty scores to [0, 1]
    train_gp : bool
        Whether to train GP hyperparameters (set False for sparse data)
    initial_sequences : List[str], optional
        Initial training sequences to populate the GP
    """

    def __init__(
        self,
        diffusion_model,
        lengthscale: float = 0.1,
        device: Optional[torch.device] = None,
        normalize_uncertainty: bool = True,
        train_gp: bool = False,  # Set False for sparse data
        initial_sequences: Optional[List[str]] = None,
        kernel: str = 'cosine',
        noise_timestep: float = 0.0,
        n_noise_samples: int = 5,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.diffusion_model = diffusion_model
        self.device = device
        self.noise_timestep = noise_timestep
        self.n_noise_samples = n_noise_samples
        self.normalize_uncertainty = normalize_uncertainty
        
        # Initialize with empty dataset
        # We'll determine feat_dim after seeing first sequences
        self.data = None
        self.feat_dim = None
        self.lengthscale = lengthscale
        self.kernel_name = kernel
        self.train_gp = train_gp
        
        self.likelihood = None
        self.gp_model = None
        
        # Initialize with training data if provided
        if initial_sequences is not None:
            print(f"Initializing GP with {len(initial_sequences)} training sequences...")
            self.add_data(initial_sequences)
            print(f"GP initialized with {len(self.data)} unique training points")
        
    def _initialize_gp(self, feat_dim: int, bootstrap_embeddings: Optional[torch.Tensor] = None):
        """Initialize GP model once we know the feature dimension."""
        self.feat_dim = feat_dim
        self.data = torch.empty(0, feat_dim, device=self.device)

        # Moderate noise for numerical stability.
        # Too-low noise (1e-4) makes the kernel matrix ill-conditioned when
        # many training embeddings have high similarity.
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
        ).to(self.device)
        self.likelihood.noise = 1e-3

        # Auto-set RBF lengthscale to mean pairwise distance of bootstrap embeddings
        # when the user passes lengthscale <= 0.
        ls = self.lengthscale
        if (
            self.kernel_name == 'rbf'
            and ls <= 0
            and bootstrap_embeddings is not None
            and bootstrap_embeddings.shape[0] >= 2
        ):
            with torch.no_grad():
                d = torch.cdist(bootstrap_embeddings, bootstrap_embeddings)
                n = d.shape[0]
                # Off-diagonal pairs only
                mask = ~torch.eye(n, dtype=torch.bool, device=d.device)
                ls = float(d[mask].mean().item())
            self.lengthscale = ls
            print(f"  [GP] auto lengthscale = {ls:.4f}  (mean pairwise distance over {n} bootstrap embeddings)")

        covar_module = build_kernel(
            self.kernel_name,
            lengthscale=ls,
            feat_dim=feat_dim,
        )
        print(f"  [GP] kernel = {self.kernel_name}  (feat_dim={feat_dim})")
        
        labels = torch.zeros(0, device=self.device)
        self.gp_model = GPModel(
            self.data, labels, self.likelihood,
            covar_module=covar_module,
        ).to(self.device)
        
        self.gp_model.eval()
        self.likelihood.eval()

    @torch.no_grad()
    def extract_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Extract embeddings from the diffusion model's backbone.
        
        Args:
            sequences: List of peptide sequence strings
            
        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        # Tokenize and encode each sequence
        encoded_list = []
        for seq in sequences:
            # First tokenize the string into tokens
            tokens = self.diffusion_model.tokenizer._tokenize(seq)
            # Then encode the tokens to get input_ids and attention_mask
            encoded = self.diffusion_model.tokenizer.encode(tokens)
            encoded_list.append(encoded)
        
        # Batch the tokenized sequences
        max_len = max(enc['input_ids'].shape[1] for enc in encoded_list)
        batch_input_ids = []
        batch_attn_mask = []
        
        for encoded in encoded_list:
            input_ids = encoded['input_ids'].squeeze(0)
            attn_mask = encoded['attention_mask'].squeeze(0)
            
            # Pad to max length
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
                attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)])
            
            batch_input_ids.append(input_ids)
            batch_attn_mask.append(attn_mask)
        
        input_ids = torch.stack(batch_input_ids).to(self.device)
        attn_mask = torch.stack(batch_attn_mask).to(self.device)
        
        # Get embeddings from backbone (RoFormer)
        self.diffusion_model.backbone.eval()
        t = self.noise_timestep
        n_draws = self.n_noise_samples if t > 0 else 1

        with torch.no_grad():
            accum = None
            for _ in range(n_draws):
                if t > 0:
                    mask_prob = torch.full(
                        (input_ids.shape[0], 1), t, device=self.device)
                    zt = self.diffusion_model.q_xt(input_ids, mask_prob)
                else:
                    zt = input_ids

                outputs = self.diffusion_model.backbone.model.roformer(
                    input_ids=zt,
                    attention_mask=attn_mask,
                    output_hidden_states=True
                )

                # Use mean pooling over sequence length (excluding padding)
                hidden_states = outputs.hidden_states[-1]  # Last layer
                mask_expanded = attn_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                emb = sum_embeddings / sum_mask

                if accum is None:
                    accum = emb
                else:
                    accum = accum + emb

            embeddings = accum / n_draws

        # L2-normalise so cosine kernel reduces to dot-product
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            
        return embeddings

    def add_data(self, sequences: List[str], max_buffer_size: Optional[int] = None):
        """
        Append new sequences to the GP training buffer.

        Args:
            sequences: peptide sequence strings to add.
            max_buffer_size: if given, after appending, keep only the most
                recent `max_buffer_size` rows (FIFO drop oldest). Used for
                rolling-window buffer mode.
        """
        # Extract embeddings
        new_embeddings = self.extract_embeddings(sequences)

        # Initialize GP if this is first data
        if self.gp_model is None:
            self._initialize_gp(new_embeddings.shape[1], bootstrap_embeddings=new_embeddings)

        # Add to dataset
        self.data = torch.cat([self.data, new_embeddings], dim=0)

        # Optional rolling-window cap: drop oldest rows so buffer stays bounded.
        if max_buffer_size is not None and self.data.shape[0] > max_buffer_size:
            self.data = self.data[-max_buffer_size:]

        # Update GP model with new data
        labels = torch.zeros(self.data.shape[0], device=self.device)
        self.gp_model.set_train_data(self.data, labels, strict=False)

        # Train the GP by optimizing hyperparameters (optional)
        if self.train_gp:
            self._train_gp()

    def replace_data(self, sequences: List[str]):
        """Replace the GP buffer with these sequences (no history retained).

        Use this to fight the buffer-feedback loop: at every AL iteration,
        refit the GP on what the *current* model is producing, so collapse
        regions immediately drop in variance instead of staying rewarded
        because they're "different from the buffer's stale snapshot."
        """
        new_embeddings = self.extract_embeddings(sequences)
        if self.gp_model is None:
            self._initialize_gp(new_embeddings.shape[1], bootstrap_embeddings=new_embeddings)
        self.data = new_embeddings
        labels = torch.zeros(self.data.shape[0], device=self.device)
        self.gp_model.set_train_data(self.data, labels, strict=False)
        if self.train_gp:
            self._train_gp()
    
    def _train_gp(self, training_iter=20):
        """
        Train GP hyperparameters using marginal log likelihood.
        Uses conservative settings to avoid overfitting with sparse data.
        """
        self.gp_model.train()
        self.likelihood.train()
        
        # Lower learning rate for stability with sparse data
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        
        prev_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.gp_model(self.data)
            loss = -mll(output, self.gp_model.train_targets)
            loss.backward()
            optimizer.step()
            
            # Early stopping if loss stops improving
            if abs(prev_loss - loss.item()) < 1e-4:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            else:
                patience_counter = 0
            prev_loss = loss.item()
        
        # Set back to eval mode
        self.gp_model.eval()
        self.likelihood.eval()

    def __call__(self, sequences: List[str], conditional: bool = False) -> np.ndarray:
        """
        Compute uncertainty scores for sequences.

        Args:
            sequences: List of peptide sequence strings
            conditional: if True, return the *conditional* posterior variance
                Var(f_i | f_{~i}, X_train) — i.e. each sample's uncertainty
                after the other samples in the same batch are also treated
                as observed. Two in-batch near-duplicates jointly inflate
                each other's denominator and both score low, so collapse
                self-penalizes. If False (default), returns the standard
                marginal posterior variance.

        Returns:
            Uncertainty scores as numpy array of shape (len(sequences),)
        """
        if self.gp_model is None:
            # No data yet - return uniform uncertainty
            return np.ones(len(sequences), dtype=np.float32)

        if conditional:
            return self.conditional_uncertainty(sequences)

        # Extract embeddings
        embeddings = self.extract_embeddings(sequences)

        # Get GP predictions
        with gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.max_root_decomposition_size(500), \
             gpytorch.settings.cholesky_jitter(1e-3):
            posterior = self.likelihood(self.gp_model(embeddings))

        uncertainty = posterior.variance.detach().cpu().numpy()

        # Normalize if requested
        if self.normalize_uncertainty and len(self.data) > 0:
            # --- Prior-variance normalization (consistent across batches) ---
            # Prior variance at any point = outputscale * k(x,x) + noise
            # For the cosine kernel k(x,x) = 1.0, so:
            prior_var = (
                self.gp_model.covar_module.outputscale.detach().cpu().item()
                + self.likelihood.noise.detach().cpu().item()
            )
            uncertainty = uncertainty / (prior_var + 1e-8)
            uncertainty = np.clip(uncertainty, 0.0, 1.0)

        return uncertainty.astype(np.float32)

    @torch.no_grad()
    def conditional_uncertainty(self, sequences: List[str],
                                jitter: float = 1e-3) -> np.ndarray:
        """Posterior variance at each x_i CONDITIONAL on the rest of the batch
        being observed alongside the GP training buffer.

        Schur-complement identity: for a joint Gaussian over (f_1, …, f_B)
        with covariance K, the conditional variance of f_i given f_{~i} is
        ``1 / [K^{-1}]_{i,i}``. We materialize the BxB posterior covariance,
        add jitter, factor it via Cholesky, then read off the diagonal of
        the inverse from the triangular solve — one factorization per call.

        Cost: O(B^3) for batch size B; ~ms for B≈2000 on GPU.
        """
        if self.gp_model is None:
            return np.ones(len(sequences), dtype=np.float32)

        embeddings = self.extract_embeddings(sequences)
        B = embeddings.shape[0]

        with gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.max_root_decomposition_size(500), \
             gpytorch.settings.cholesky_jitter(1e-3):
            posterior = self.likelihood(self.gp_model(embeddings))
            K = posterior.covariance_matrix  # (B, B) — includes noise on diag

        eye = torch.eye(B, device=K.device, dtype=K.dtype)
        K = K + jitter * eye

        # diag(K^{-1}) via triangular solve: L L^T = K  ⇒  K^{-1} = (L^{-T})(L^{-1})
        # so diag(K^{-1})_i = sum_k (L^{-1})_{k,i}^2.
        L = torch.linalg.cholesky(K)
        X = torch.linalg.solve_triangular(L, eye, upper=False)
        diag_K_inv = (X ** 2).sum(dim=0).clamp(min=1e-12)
        cond_var = 1.0 / diag_K_inv

        cond_var = cond_var.detach().cpu().numpy()

        if self.normalize_uncertainty and len(self.data) > 0:
            prior_var = (
                self.gp_model.covar_module.outputscale.detach().cpu().item()
                + self.likelihood.noise.detach().cpu().item()
            )
            cond_var = cond_var / (prior_var + 1e-8)
            cond_var = np.clip(cond_var, 0.0, 1.0)

        return cond_var.astype(np.float32)