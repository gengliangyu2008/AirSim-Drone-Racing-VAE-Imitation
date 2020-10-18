import time
#import datetime
from datetime import datetime
from time import gmtime,strftime

if __name__ == '__main__':
    # print(strftime("%Y-%m-%d %H:%M:%S", time()))
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
