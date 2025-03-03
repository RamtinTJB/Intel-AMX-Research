def encoding_rp_torch(X_data, base_matrix, signed=False):
    """
    PyTorch based RP encoding that always uses bfloat 16 autocast
    """
    import torch.amp  # Make sure you have a recent PyTorch

    # Convert arrays to Torch float32
    A = torch.from_numpy(base_matrix).float()  # shape (D, n_features)
    X_arr = np.array(X_data, dtype=np.float32)
    X_torch = torch.from_numpy(X_arr)  # shape (n_samples, n_features)

    enc_hvs = []
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        for i in range(len(X_torch)):
            # Optional progress
            if i % max(int(len(X_torch) / 20), 1) == 0:
                sys.stdout.write(f"{int(i/len(X_torch)*100)}% ")
                sys.stdout.flush()

            hv_torch = torch.matmul(A, X_torch[i])  # shape (D,)
            hv_np = hv_torch.cpu().float().numpy()  # back to NumPy float

            enc_hvs.append(hv_np)

    return enc_hvs

def encoding_idlv_torch(X_data, lvl_hvs, id_hvs, D, bin_len, x_min, L=64):
    """
    PyTorch-based IDLV encoding (with AMX if available).
    """
    import torch.amp

    lvl_hvs_torch = torch.from_numpy(lvl_hvs).float()  # (L, D)
    id_hvs_torch  = torch.from_numpy(id_hvs).float()   # (n_features, D)
    X_arr = np.array(X_data, dtype=np.float32)
    X_torch = torch.from_numpy(X_arr)

    x_min_torch   = torch.tensor(x_min,   dtype=torch.float32)
    bin_len_torch = torch.tensor(bin_len, dtype=torch.float32)

    enc_hvs = []
    n_samples = X_torch.shape[0]

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        for i in range(n_samples):
            if i % max(int(n_samples / 20), 1) == 0:
                sys.stdout.write(f"{int(i/n_samples*100)}% ")
                sys.stdout.flush()

            x = X_torch[i]
            bins = torch.floor((x - x_min_torch)/bin_len_torch).to(torch.int64)
            bins = torch.clamp(bins, 0, L-1)

            chosen_lvl = lvl_hvs_torch[bins]           # shape = (n_features, D)
            hv_torch = (chosen_lvl * id_hvs_torch).sum(dim=0)

            hv_np = hv_torch.cpu().float().numpy()
            enc_hvs.append(hv_np)

    return enc_hvs
