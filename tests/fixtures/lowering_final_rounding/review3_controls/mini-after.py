class Helper:
    def intp(self, x): return x+1
helper=Helper()
def f():
    s=0
    for i in range(3):
        i=helper.intp(i)
        s+=i
    return s
