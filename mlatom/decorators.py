from . import data


def predict_wrapper(predict):
    def wrapper(
        self,
        molecular_database: data.molecular_database = None,
        molecule: data.molecule = None,
        *args,
        **kwargs
    ):
        self.set_num_threads()
        if molecular_database != None:
            molecular_database = molecular_database
        elif molecule != None:
            molecular_database = data.molecular_database([molecule])
        else:
            errmsg = "Either molecule or molecular_database should be provided in input"
            raise ValueError(errmsg)
        
        result = predict(self, molecular_database, *args, **kwargs)
        self.unset_num_threads()
        return result
    
    return wrapper


class doc_inherit:
    """
    Docstring inheriting (and append) method descriptor
    
    The class itself is also used as a decorator
    
    Modified from `https://code.activestate.com/recipes/576862/`_
    """
    
    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__
    
    def __get__(self, obj, cls):
        docs = [self.mthd.__doc__ if self.mthd.__doc__ else ""]
        for parent in cls.__mro__[1:]:
            parent_mthd = getattr(parent, self.name, None)
            if parent_mthd:
                if parent_mthd.__doc__:
                    docs.append(parent_mthd.__doc__)
                break
        docs.reverse()
        from functools import wraps
        
        @wraps(self.mthd, assigned=("__name__", "__module__", "__doc__"))
        def func(*args, **kwargs):
            if obj:
                return self.mthd(obj, *args, **kwargs)
            else:
                return self.mthd(*args, **kwargs)
        
        func.__doc__origin__ = func.__doc__
        func.__doc__ = "\n".join([doc.rstrip() for doc in docs])
        return func
