

def log(prefix, action_name):
    def log_decorator(f):
        def wrapper(*args, **kw):
            if(prefix == "DEBUG"):
                print("############### "+ action_name +" start #################")
            f(*args, **kw)
            if(prefix == "DEBUG"):
                print("############### " +action_name+ " end #################\n")
        return wrapper
    return log_decorator

@log('DEBUG',"read astropy map")
def test(a,b=2):
    print(a,b)
    return 1