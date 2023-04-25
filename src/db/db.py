def init_data_set(mydb, mycursor):
    # 创建数据集data_set表格
    mycursor.execute("""
        CREATE TABLE IF NOT EXISTS data_set (
            id INT AUTO_INCREMENT PRIMARY KEY,
            data_set_name VARCHAR(255),
            data_set_desc TEXT,
            image_count INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 创建data_images表格
    mycursor.execute("""
        CREATE TABLE IF NOT EXISTS data_images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            data_set_id INT,
            image_path VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (data_set_id) REFERENCES data_set(id)
        )
    """)
    mydb.commit()
    print('创建表格成功')