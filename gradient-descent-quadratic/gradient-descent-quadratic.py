def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # f = a^2 * x + b * x + c
    
    for i in range(steps): 
        # Calculating f's derivative
        f_der = 2 * a * x0 + b

        # Updating the x_t every step
        x0 = x0 - lr * f_der # 1st x0 in this line == x at t+1, 2nd one is x at t.
    
    return x0
    