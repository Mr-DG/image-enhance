import mysql.connector
from .db import init_data_set

# 数据库连接参数
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'image_enhance'
}
mydb = mysql.connector.connect(**db_config)
mycursor = mydb.cursor()

# 链接数据库
def mysql_connect():
    mycursor.execute("CREATE DATABASE IF NOT EXISTS image_enhance")
    print('数据库链接成功')
    init_dB(mydb, mycursor)

# 初始化数据库表
def init_dB(mydb, mycursor):
    init_data_set(mydb, mycursor)
