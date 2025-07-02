# setup.py
from setuptools import setup, find_packages

setup(
	name="jodalroB_common",  # 패키지 이름 (pip에 표시될 이름)
	version="0.1.0",
	description="Common utilities for jodalroB-ML project",
	packages=find_packages(include=["common", "common.*"]),
	install_requires=[  # 필요하다면 여기에 의존 라이브러리 나열
		# e.g. "sshtunnel", "sqlalchemy>=1.4"
	],
	python_requires=">=3.8",
)
