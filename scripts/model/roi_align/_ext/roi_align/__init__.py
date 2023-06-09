
# from torch.utils.ffi import _wrap_function
# from ._roi_align import lib as _lib, ffi as _ffi

# __all__ = []
# def _import_symbols(locals):
#     for symbol in dir(_lib):
#         fn = getattr(_lib, symbol)
#         if callable(fn):
#             locals[symbol] = _wrap_function(fn, _ffi)
#         else:
#             locals[symbol] = fn
#         __all__.append(symbol)

# _import_symbols(locals())

import torch.ops._extroialign as _lib

__all__ = [symbol for symbol in dir(_lib) if not symbol.startswith('_')]

def _import_symbols(locals):
    for symbol in __all__:
        locals[symbol] = getattr(_lib, symbol)

_import_symbols(locals())