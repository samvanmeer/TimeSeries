from datetime import datetime
fmt='%Y-%m-%d'
def sformat(date):
    return date.strftime(fmt)

def dformat(date):
    return datetime.strptime(date,fmt)