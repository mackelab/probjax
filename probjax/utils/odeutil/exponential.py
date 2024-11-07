

# def exponential_euler(drift, t0, y0, f0, dt):
#     jacobian_fn = jax.jacfwd(drift, argnums=1)

#     A = jacobian_fn(t0, y0)
#     B = jnp.zeros_like(A)
#     C = jnp.eye(A.shape[0])
#     H = jnp.block([[A, C], [B, B]])
#     eHdt = jax.scipy.linalg.expm(H * dt)
#     phi0 = eHdt[0 : A.shape[0], 0 : A.shape[1]]
#     phi1 = eHdt[0 : A.shape[0], A.shape[1] :]

#     y1 = phi0 @ y0 + dt * phi1 @ (f0 - A @ y0)
#     f1 = drift(t0 + dt, y1)

#     return y1, f1, None


# info = {
#     "explicit": False,
#     "order": 2,
#     "info": "Exponential Euler method",
#     "adaptive": False,
# }

# register_method("exp_euler", exponential_euler, info=info)
