# TODO: Implement this class

# class DiffusionScoreMatcher(nnx.Module, experimental_pytree=True):
#     def __init__(
#         self,
#         net: nnx.Module,
#         std0: ArrayLike = 1.0,
#         scale_fn: Optional[Callable] = None,
#         std_fn: Optional[Callable] = None,
#         rngs=None,
#     ):
#         self.net = net
#         if scale_fn is not None:
#             self.scale_fn = scale_fn
#         if std_fn is not None:
#             self.std_fn = std_fn
#         self.std0 = nnx.Variable(std0)

#     def c_in(self, t):
#         return 1.0

#     def c_out(self, t):
#         return 1.0

#     def c_t(self, t):
#         return self.std_fn(t)

#     def c_skip(self, t):
#         return None

#     def weight_fn(self, t):
#         return 1.0
