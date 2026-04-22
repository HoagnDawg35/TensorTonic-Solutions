import numpy as np

def rmsprop_step(w, g, s, lr=0.1, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # input_name = ["w", "g", "s"]
    # for i in range(len(input_name)):
    #     out = np.asarray(input_name[i], dtype=float)
    #     if i == 0:
    #         w = out
    #     elif i == 1:
    #         g = out
    #     elif i == 2:
    #         s = out

    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    
    # Update running avg
    s = (beta * s) + ((1-beta) * (g*g))

    # Params update
    # w = w - lr * (beta / (np.sqrt(s + eps) )) * g
    numerator = lr * g
    denominator = np.sqrt(s + eps)

    w = w - numerator / denominator

    return (w, s)

# w = [1.0, 2.0]
# g = [0.2, -0.4]
# s = [0, 0]

# w, s = rmsprop_step(w, g, s)
# print(w, s)