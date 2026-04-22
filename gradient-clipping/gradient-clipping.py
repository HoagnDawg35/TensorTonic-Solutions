import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g, dtype=float)

    g_norm = np.linalg.norm(g) # Calculating g_norm

    if max_norm > 0: # Handle max_norm <= 0 cases
        if g_norm <= max_norm:
            g = g.copy()
        else:
            g = g.copy() * max_norm / g_norm
    else:
        return g.copy() # return g-shaped 0
    
    return g
    