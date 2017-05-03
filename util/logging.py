import logging

# 定义日志格式，并且设置日志文件的名称
def setup_logging():
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='log.txt', filemode='w',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO, format=FORMAT)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)