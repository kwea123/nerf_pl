import torch
from einops import rearrange, reduce, repeat

__all__ = ['render_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                ts,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays and @ts
    Inputs:
        models: dict of NeRF models (coarse and fine) defined in nerf.py
        embeddings: dict of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3), ray origins and directions
        ts: (N_rays), ray time as embedding index
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points on each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        """
        typ = model.typ
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c', c=3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ=='coarse' and test_time:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                out_chunks += [model(xyz_embedded, sigma_only=True)]
            out = torch.cat(out_chunks, 0)
            static_sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            # create other necessary inputs
            if model.encode_appearance:
                a_embedded_ = repeat(a_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            if output_transient:
                t_embedded_ = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            for i in range(0, B, chunk):
                # inputs for original NeRF
                inputs = [embedding_xyz(xyz_[i:i+chunk]), dir_embedded_[i:i+chunk]]
                # additional inputs for NeRF-W
                if model.encode_appearance:
                    inputs += [a_embedded_[i:i+chunk]]
                if output_transient:
                    inputs += [t_embedded_[i:i+chunk]]
                out_chunks += [model(torch.cat(inputs, 1), output_transient=output_transient)]

            out = torch.cat(out_chunks, 0)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_)
            static_rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            static_sigmas = out[..., 3] # (N_rays, N_samples_)
            if output_transient:
                transient_rgbs = out[..., 4:7]
                transient_sigmas = out[..., 7]
                transient_betas = out[..., 8]

        # Convert these values using volume rendering
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        if output_transient:
            static_alphas = 1-torch.exp(-deltas*static_sigmas)
            transient_alphas = 1-torch.exp(-deltas*transient_sigmas)
            alphas = 1-torch.exp(-deltas*(static_sigmas+transient_sigmas))
        else:
#             noise = torch.randn_like(static_sigmas) * noise_std
            alphas = 1-torch.exp(-deltas*static_sigmas)

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1) # [1, 1-a1, 1-a2, ...]
        transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]

        if output_transient:
            static_weights = static_alphas * transmittance
            transient_weights = transient_alphas * transmittance

        weights = alphas * transmittance
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        if output_transient:
            results['transient_sigmas'] = transient_sigmas
        if test_time and typ == 'coarse':
            return


        if output_transient:
            static_rgb_map = reduce(rearrange(static_weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                                    'n1 n2 c -> n1 c', 'sum')
            if white_back:
                static_rgb_map += 1-rearrange(weights_sum, 'n -> n 1')
            
            transient_rgb_map = \
                reduce(rearrange(transient_weights, 'n1 n2 -> n1 n2 1')*transient_rgbs,
                       'n1 n2 c -> n1 c', 'sum')
            results['beta'] = reduce(transient_weights*transient_betas, 'n1 n2 -> n1', 'sum')
            # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
            # See "Notes on differences with the paper" in README.
            results['beta'] += model.beta_min
            
            # the rgb maps here are when both fields exist
            results['_rgb_fine_static'] = static_rgb_map
            results['_rgb_fine_transient'] = transient_rgb_map
            results['rgb_fine'] = static_rgb_map + transient_rgb_map

            if test_time:
                # Compute also static and transient rgbs when only one field exists.
                # The result is different from when both fields exist, since the transimttance
                # will change.
                static_alphas_shifted = \
                    torch.cat([torch.ones_like(static_alphas[:, :1]), 1-static_alphas], -1)
                static_transmittance = torch.cumprod(static_alphas_shifted[:, :-1], -1)
                static_weights_ = static_alphas * static_transmittance
                static_rgb_map_ = \
                    reduce(rearrange(static_weights_, 'n1 n2 -> n1 n2 1')*static_rgbs,
                           'n1 n2 c -> n1 c', 'sum')
                if white_back:
                    static_rgb_map_ += 1-rearrange(weights_sum, 'n -> n 1')
                results['rgb_fine_static'] = static_rgb_map_
                results['depth_fine_static'] = \
                    reduce(static_weights_*z_vals, 'n1 n2 -> n1', 'sum')

                transient_alphas_shifted = \
                    torch.cat([torch.ones_like(transient_alphas[:, :1]), 1-transient_alphas], -1)
                transient_transmittance = torch.cumprod(transient_alphas_shifted[:, :-1], -1)
                transient_weights_ = transient_alphas * transient_transmittance
                results['rgb_fine_transient'] = \
                    reduce(rearrange(transient_weights_, 'n1 n2 -> n1 n2 1')*transient_rgbs,
                           'n1 n2 c -> n1 c', 'sum')
                results['depth_fine_transient'] = \
                    reduce(transient_weights_*z_vals, 'n1 n2 -> n1', 'sum')
        else: # no transient field
            rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                             'n1 n2 c -> n1 c', 'sum')
            if white_back:
                rgb_map += 1-rearrange(weights_sum, 'n -> n 1')
            results[f'rgb_{typ}'] = rgb_map

        results[f'depth_{typ}'] = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')
        return

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d))

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    results = {}
    output_transient = False
    inference(results, models['coarse'], xyz_coarse, z_vals, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here
        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        model = models['fine']
        if model.encode_appearance:
            if 'a_embedded' in kwargs:
                a_embedded = kwargs['a_embedded']
            else:
                a_embedded = embeddings['a'](ts)
        output_transient = kwargs.get('output_transient', True) and model.encode_transient
        if output_transient:
            if 't_embedded' in kwargs:
                t_embedded = kwargs['t_embedded']
            else:
                t_embedded = embeddings['t'](ts)
        inference(results, model, xyz_fine, z_vals, test_time, **kwargs)

    return results
