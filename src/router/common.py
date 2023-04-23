import json
import os
from flask import Blueprint, request, current_app, Flask
from werkzeug.utils import secure_filename
import uuid

common = Blueprint('common', __name__)


# 登录
@common.route('/login', methods=['POST'])
def login():

    obj = json.loads(request.data)
    return {
        'code': 200,
        'msg': '操作成功',
        'data': obj
    }


# 文件上传
@common.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return '文件字段为file'

    UPLOAD_FOLDER = current_app.config['UPLOAD_FOLDER']

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # 文件夹不存在则生成文件夹
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # 生成一个随机字符串作为文件名的一部分
    random_str = str(uuid.uuid4())

    file.save(os.path.join(
        UPLOAD_FOLDER, secure_filename(random_str + '_' + file.filename)))

    url = request.host_url + UPLOAD_FOLDER + random_str + '_' + file.filename

    return {
        'code': 200,
        'msg': '上传成功',
        'url': url
    }
