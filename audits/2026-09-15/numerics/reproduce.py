import jax
import jax.numpy as jnp
from scipy.stats import differential_entropy as reference_entropy
from probjax.utils.stats import differential_entropy
from probjax.utils.linalg import batched_pcg_solve, lanczos_logdet, is_triangular_matrix, is_diagonal_matrix
print('JAX',jax.__version__)
def check(name,fn):
 try: print(name,fn())
 except Exception as e: print(name,type(e).__name__,str(e)[:180])
x=jnp.linspace(.001,.999,100)
for method in ['vasicek','van es','ebrahimi','correa','auto']:
 check('entropy '+method,lambda method=method:(differential_entropy(x,method=method),reference_entropy(x,method=method)))
for scale in [1.,1e-4]:
 check('PCG identity scale '+str(scale),lambda scale=scale:batched_pcg_solve(lambda x:x,jnp.ones((2,1))*scale,block_size=1,tol=1e-5))
check('logdet [[2]] expected .693147',lambda:lanczos_logdet(jnp.array([[2.]]),num_steps=1))
check('upper triangle expected True',lambda:is_triangular_matrix(jnp.array([[1.,2.],[0.,1.]]),lower=False))
check('batch diagonal expected [True True]',lambda:is_diagonal_matrix(jnp.stack([jnp.eye(2),2*jnp.eye(2)])))
S=jnp.array([[2.,1.],[1.,2.]])
check('full-step SLQ expected log3',lambda:(lanczos_logdet(S,num_steps=2),jnp.linalg.slogdet(S)[1]))
