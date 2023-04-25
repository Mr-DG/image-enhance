import json
import os
from flask import Blueprint, request, current_app, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import zipfile
from src.db.connect import mydb, mycursor

dataset = Blueprint('dataset', __name__)

# zip文件上传
@dataset.route('/upload_zip', methods=['POST'])
def upload_zip():
    # 检查上传的文件是否为zip格式
    if 'file' not in request.files:
        return '文件字段为file'

    UPLOAD_FOLDER = current_app.config['UPLOAD_FOLDER']

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    # 文件夹不存在则生成文件夹
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if file and file.filename.endswith('.zip') or file.filename.endswith('.rar'):
        filename = file.filename
        # 生成一个随机字符串作为文件名的一部分
        random_str = str(uuid.uuid4())
        filepath = os.path.join(
            UPLOAD_FOLDER, secure_filename(random_str + '_' + filename))
        file.save(filepath)
        return {
            'code': 200,
            'msg': '上传成功',
            'filename': random_str + '_' + filename
        }
    else:
        return '失败'


# 保存数据集
@dataset.route('/save_data_set', methods=['POST'])
def save_data_set():
    # 获取请求参数
    obj = json.loads(request.data)
    data_set_name = obj['dataSetName']
    data_set_desc = obj['dataSetDesc']
    zip_name = obj['zipName']


    # 获取zip文件路径
    zip_path = os.path.join('upload', zip_name)
    
    # 判断zip文件是否存在
    if not os.path.exists(zip_path):
        print('文件不存在')
        return {'message': 'Zip file does not exist!'}

    # 解压后保存的文件夹名
    DATA_SET_IMG = current_app.config['DATA_SET_IMG']

    # 创建img文件夹
    os.makedirs(DATA_SET_IMG, exist_ok=True)

    # 数据集对应的图片路径
    images_path = []
    # 解压缩zip文件
    zip_file = zipfile.ZipFile(zip_path)
    if zip_file:
        for names in zip_file.namelist():
            # 如果文件是图片类型，则保存到img文件夹下
            if names.endswith(('.jpg', '.png', '.bmp')):
                zip_file.extract(names, DATA_SET_IMG)
                images_path.append(f'{request.host_url}{DATA_SET_IMG}/{names}')
                           
    # 保存数据集信息到data_set表
    insert_data_set_query = '''
        INSERT INTO data_set (data_set_name, data_set_desc, image_count)
        VALUES (%s, %s, %s)
    '''

    mycursor.execute(insert_data_set_query, (data_set_name, data_set_desc, len(images_path)))
    mydb.commit()

    
    # 获取数据集id
    data_set_id = mycursor.lastrowid

    # 保存数据集图片路径到data_images表
    insert_data_images_query = '''
        INSERT INTO data_images (data_set_id, image_path)
        VALUES (%s, %s)
    '''
    for image_path in images_path:
        mycursor.execute(insert_data_images_query, (data_set_id, image_path))
        mydb.commit()
    
    return {
        'code': 200,
        'msg': '操作成功'
    }
