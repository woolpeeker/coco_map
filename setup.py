from setuptools import setup, find_packages

setup(
    name="coco_map",
    version="0.1",
    author="JiapengLuo",
    author_email="luojiapeng1993@gmail.com",
    description="A tools for calculating coco mAP",

    # 项目主页
    url="https://github.com/woolpeeker/coco_map",

    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages()
)