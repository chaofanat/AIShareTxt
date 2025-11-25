#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIShareTxt包安装配置
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def read_requirements(filename):
    """读取requirements文件"""
    requirements = []
    try:
        with open(filename, "r", encoding="utf-8") as fh:
            content = fh.read()
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    except FileNotFoundError:
        pass
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(filename, "r", encoding="gbk") as fh:
                content = fh.read()
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        requirements.append(line)
        except (FileNotFoundError, UnicodeDecodeError):
            pass
    return requirements

# 核心依赖
install_requires = read_requirements("requirements.txt")

# 开发依赖
dev_requires = read_requirements("requirements-dev.txt")

setup(
    name="aishare-txt",
    version="2025.11.24.26",
    author="AIShareTxt Team",
    author_email="chaofanat@gmail.com",
    description="中国股票技术指标文本生成工具包，用于为金融分析相关领域的AI智能体提供上下文服务。",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitee.com/chaofanat/aishare-txt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    license="MIT",
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "ai": ["openai>=1.0.0", "zhipuai>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "aishare=AIShareTxt.core.data_processor:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)