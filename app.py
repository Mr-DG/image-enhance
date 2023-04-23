from flask import Flask, send_from_directory
from src.router.common import common

app = Flask(__name__)
app.register_blueprint(common, url_prefix='/')


# hello world
@app.route('/')
def hello():
    print(app.config['UPLOAD_FOLDER'])
    return 'Hello, World!'


# 文件上传路径
# 托管静态资源
app.config['UPLOAD_FOLDER'] = 'upload/'
app.static_folder = 'static'


@app.route('/static/<path:filename>')
def send_image(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/upload/<path:filename>')
def send_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if (__name__ == '__main__'):
    app.run('127.0.0.1', 8090, debug=True)  # debug=True 热更新
