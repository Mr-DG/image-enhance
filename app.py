from flask import Flask
from src.router.common import common
from src.router.data_set import dataset
from src.db.connect import mysql_connect
from src.router.enhance import enhance

app = Flask(__name__)

# router
app.register_blueprint(common, url_prefix='/') # 公共路由
app.register_blueprint(dataset, url_prefix='/dataset') # 数据集路由
app.register_blueprint(enhance, url_prefix='/enhance') # 数据集路由

# 链接数据库
mysql_connect()

# 托管静态资源路径
app.config['UPLOAD_FOLDER'] = 'upload/' # 上传文件夹
app.static_folder = 'static' # 静态资源文件夹
app.config['DATA_SET_IMG'] = 'data_set_img' # 数据集解压之后图片保存的文件夹

if (__name__ == '__main__'):
    app.run('127.0.0.1', 8090, debug=True)  # debug=True 热更新
