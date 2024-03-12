# -*- coding=utf-8
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
import logging
import zlib
import hashlib
from qcloud_cos import CosClientError
from qcloud_cos import CosServiceError

# 正常情况日志级别使用 INFO，需要定位时可以修改为 DEBUG，此时 SDK 会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# https://aishow-1315251136.cos.ap-hongkong.myqcloud.com/ai_show/4JY69LklCTRmXoV9.png?q-sign-algorithm=sha1&q-ak=AKID27iaF_HhBNIFPiKp78MAKor6TrOOxXO4b1Q5BN-DXXHSFvhBMpUbWnQckL6xr2Z0&q-sign-time=1692953850;1692957450&q-key-time=1692953850;1692957450&q-header-list=host&q-url-param-list=ci-process&q-signature=b889ba378d086f099950aad95ab7f396af870639&x-cos-security-token=0Iq4Pl7AKbeAcJzROLwjp5XqeGPX6zma7cfd110c7fa74ce53544eb31c604a3c3pyqjVq5EG0bfDfmOYNpTVI_Y6t_pv-5AlP0z42LktrVxieh6xYg-JNTmO8zZ7b50BfJAdgFdVROGwOCtY0-V7DC7vT00M6nA5VpK-onM4LUytO1D9oUT-Raw7zCd37anLEp0yd-3ZlY6UZxxkeJRFZ74StYY4zeUTYK6tPQWcomAa_fye5I1XkVAQ8IaoAG0&ci-process=originImage
# 1. 设置用户属性, 包括 secret_id, secret_key, region等。Appid 已在 CosConfig 中移除，请在参数 Bucket 中带上 Appid。Bucket 由 BucketName-Appid 组成
secret_id = 'AKIDiGafcOwFJ9lbCajdIiusa4KLbgaGiwwk'    # 用户的 SecretId，建议使用子账号密钥，授权遵循最小权限指引，降低使用风险。子账号密钥获取可参见 https://cloud.tencent.com/document/product/598/37140
secret_key = 'd1B6nhcMwMbdS4WnskQt9D7qNh3DD6tX'  # 用户的 SecretKey，建议使用子账号密钥，授权遵循最小权限指引，降低使用风险。子账号密钥获取可参见 https://cloud.tencent.com/document/product/598/37140
region = 'ap-hongkong'      # 替换为用户的 region，已创建桶归属的 region 可以在控制台查看，https://console.cloud.tencent.com/cos5/bucket
                           # COS 支持的所有 region 列表参见 https://cloud.tencent.com/document/product/436/6224

BucketName = 'aishow-1315251136'
token = None               # 如果使用永久密钥不需要填入 token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见 https://cloud.tencent.com/document/product/436/14048
scheme = 'https'           # 指定使用 http/https 协议来访问 COS，默认为 https，可不填


config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
client = CosS3Client(config)

class TosService:
    @staticmethod
    def upload_percentage(consumed_bytes, total_bytes):
        """进度条回调函数，计算当前上传的百分比


        :param consumed_bytes: 已经上传的数据量
        :param total_bytes: 总数据量
        """
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            print('\r{0}% '.format(rate))
            sys.stdout.flush()

    @staticmethod
    def upload_file(filePath, fileKey, bucketName=BucketName):
        """上传文件到 COS

        :param filePath: 本地文件路径
        """
        # response = client.upload_file(
        #     Bucket=bucketName,
        #     Key=fileKey,
        #     LocalFilePath=filePath,
        #     progress_callback=TosService.upload_percentage,
        # )

        for i in range(0, 10):
            try:
                response = client.upload_file(
                Bucket=bucketName,
                Key=fileKey,
                LocalFilePath=filePath)
                break
            except CosClientError or CosServiceError as e:
                print(e)
                print(f"上传文件失败，重试{i}")
                continue


        print(f"文件上传完毕：{response}")

    @staticmethod
    def fileMd5(filePath):
        hash_md5 = hashlib.md5()
        with open(filePath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        fileHash = hash_md5.hexdigest()
        # print(fileHash)
        return fileHash 